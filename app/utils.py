import typing

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, trim_messages
from mem0 import Memory

from app.config import settings


def load_chat_model(
    fully_specified_name: str, temperature: int = 0, max_tokens: int = 2048
) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
        temperature (int, optional): Temperature parameter. Defaults to 0.
        max_tokens (int, optional): Max number of tokens. Defaults to 2048.
    """
    provider, model = fully_specified_name.split(":", maxsplit=1)
    return init_chat_model(
        model, model_provider=provider, temperature=temperature, max_tokens=max_tokens
    )


def load_embeddings_model() -> Embeddings:
    return typing.cast(Embeddings, init_embeddings(settings.embeddings_model))


def get_memory() -> Memory:
    config = {"version": "v1.1"}
    return Memory.from_config(config)


def trim_agent_messages(
    messages: typing.Sequence[BaseMessage], max_tokens: int = 10
) -> typing.Sequence[BaseMessage]:
    return typing.cast(
        typing.Sequence[BaseMessage],
        trim_messages(
            messages,
            max_tokens=max_tokens,
            strategy="last",
            token_counter=len,
            # Usually, we want to keep the SystemMessage
            # if it's present in the original history.
            # The SystemMessage has special instructions for the model.
            include_system=True,
            # Most chat models expect that chat history starts with either:
            # (1) a HumanMessage or
            # (2) a SystemMessage followed by a HumanMessage
            # start_on="human" makes sure we produce a valid chat history
            start_on="human",
            end_on=("human", "tool", "ai"),
        ),
    )


def prepare_recall_memory(recall_memories: list[str] | None = None) -> str:
    recall_str = ""
    if recall_memories:
        recall_str = (
            "<recall_memory>\n" + "\n".join(recall_memories) + "\n</recall_memory>"
        )
    return recall_str


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def process_recall_memory(
    memories: dict[str, typing.Any], threshold: float = 0.3
) -> list[str]:
    return [
        memory["memory"]
        for memory in memories["results"]
        if memory["score"] >= threshold
    ]


def prepare_memory_messages(
    messages: typing.Sequence[BaseMessage],
) -> list[dict[str, str]]:
    type_role_map = {"ai": "assistant", "human": "user"}

    return [
        {
            "role": type_role_map.get(msg.type, "assistant"),
            "content": get_message_text(msg),
        }
        for msg in messages
        if isinstance(msg, (HumanMessage, AIMessage)) and msg.content
    ]
