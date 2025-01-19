import logging
import typing
from datetime import datetime

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import ToolNode
from mem0 import Memory

from app import prompts, schemas, utils
from app.config import AgentConfiguration, settings
from app.tools import AgentToolkit

logger = logging.getLogger(__name__)


class Agent:
    def __init__(
        self, llm: BaseChatModel, checkpoint: BaseCheckpointSaver[str], memory: Memory
    ):
        self._llm = llm
        self._checkpoint = checkpoint
        self._memory_store = memory
        self._graph = self._setup_graph()

    async def stream(
        self, message: str, config: RunnableConfig
    ) -> typing.AsyncIterator[str]:
        inputs = {
            "messages": [HumanMessage(content=message)],
            "today": datetime.now().isoformat(),
        }
        async for event in self._graph.astream_events(
            inputs,
            config=config,
            version="v2",
        ):
            kind = event["event"]
            metadata = event["metadata"]
            if kind != "on_chat_model_stream" or metadata["langgraph_node"] != "agent":
                continue
            chunk = event["data"]["chunk"]
            if not isinstance(chunk, AIMessage):
                continue
            if not chunk.content:
                continue

            yield utils.get_message_text(chunk)

    async def invoke(self, message: str, config: RunnableConfig) -> str:
        inputs = {
            "messages": [HumanMessage(content=message)],
            "today": datetime.now().isoformat(),
        }
        reply = await self._graph.ainvoke(inputs, config)
        return utils.get_message_text(reply["messages"][-1])

    def _setup_graph(self) -> CompiledGraph:
        agent_tools = AgentToolkit(llm=self._llm).get_tools()

        system_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompts.AGENT_PROMPT),
                ("system", "Today is {today}"),
                ("placeholder", "{messages}"),
            ]
        )
        self._model = system_prompt | self._llm.bind_tools(agent_tools)

        tool_node = ToolNode(agent_tools)
        workflow = StateGraph(schemas.State)
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", tool_node)
        workflow.add_node("save_memories", self._save_memories)
        # Set the entrypoint as `agent`
        # This means that this node is the first one called
        workflow.set_entry_point("agent")
        # We now add a conditional edge
        workflow.add_conditional_edges(
            # First, we define the start node. We use `agent`.
            # This means these are the edges taken after the `agent` node is called.
            "agent",
            self._should_continue,
        )
        workflow.add_edge("tools", "agent")
        workflow.set_finish_point("save_memories")

        graph = workflow.compile(checkpointer=self._checkpoint, debug=settings.debug)
        graph_bytes = graph.get_graph().draw_mermaid_png()

        file_name = settings.graphs_dir / "graph.png"
        with open(file_name, "wb") as f:
            f.write(graph_bytes)
        logger.info("Saved agent graph to %s", file_name)

        return graph

    def _call_model(
        self, state: schemas.State, config: RunnableConfig
    ) -> dict[str, typing.Sequence[BaseMessage]]:
        messages = utils.trim_agent_messages(state["messages"])

        response = self._model.invoke(
            {
                "messages": messages,
                "today": datetime.now().isoformat(),
            },
            config,
        )
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

    @staticmethod
    def _should_continue(
        state: schemas.State,
    ) -> typing.Literal["tools", "save_memories"]:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
            return "tools"
        return "save_memories"

    def _save_memories(self, state: schemas.State, config: RunnableConfig) -> None:
        cfg = AgentConfiguration.from_runnable_config(config)
        messages = utils.prepare_memory_messages(state["messages"])

        self._memory_store.add(messages, user_id=cfg.user_id)

    def get_state(self, user_id: str, thread_id: str) -> dict[str, typing.Any]:
        config = RunnableConfig(
            configurable=dict(thread_id=thread_id, user_id=user_id),
        )
        return self._graph.get_state(config).values
