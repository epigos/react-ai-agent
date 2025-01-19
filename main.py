import logging
import typing

import chainlit as cl
from chainlit import ChatSettings, input_widget
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from mem0 import Memory

from app import logging_config, utils
from app.agent import Agent
from app.utils import load_chat_model

logging_config.configure()

logger = logging.getLogger(__name__)


def get_settings() -> ChatSettings:
    return cl.ChatSettings(
        [
            input_widget.Select(
                id="Model",
                label="LLM - Model",
                values=[
                    "openai:gpt-4o",
                    "anthropic:claude-3-5-sonnet-20241022",
                    "openai:gpt-4o-mini",
                ],
                initial_index=0,
            ),
            input_widget.Slider(
                id="MaximumTokens",
                label="Model - Maximum Token",
                initial=1024,
                min=64,
                max=4096,
                step=64,
            ),
            input_widget.Slider(
                id="Temperature",
                label="Model - Temperature",
                initial=0.2,
                min=0,
                max=2,
                step=0.1,
            ),
        ]
    )


@cl.cache
def get_checkpoint() -> BaseCheckpointSaver[str]:
    return MemorySaver()


@cl.cache
def get_memory() -> Memory:
    return utils.get_memory()


@cl.on_settings_update
async def setup_agent(chat_settings: dict[str, typing.Any]) -> None:
    logger.info("Setting up agent with following settings:\n %s", chat_settings)
    llm_model = load_chat_model(
        fully_specified_name=chat_settings["Model"],
        temperature=chat_settings["Temperature"],
        max_tokens=chat_settings["MaximumTokens"],
    )
    checkpoint_memory = cl.user_session.get(
        "checkpoint_memory", default=get_checkpoint()
    )
    cl.user_session.set("model", chat_settings["Model"])

    memory = cl.user_session.get("memory", default=get_memory())

    agent = Agent(llm_model, checkpoint_memory, memory)
    cl.user_session.set("agent", agent)


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("memory", get_memory())
    chat_settings = await get_settings().send()
    await setup_agent(chat_settings)


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Shipping times",
            message="What is the estimated shipping date from now if I make the order?",
        ),
        cl.Starter(
            label="Order history",
            message="What is my order history?",
        ),
        cl.Starter(
            label="Open Account",
            message="I would like to open an account.",
        ),
        cl.Starter(
            label="Book appointment",
            message="I'd to book an appointment.",
        ),
    ]


@cl.on_message
async def main(message: cl.Message) -> None:
    agent = typing.cast(Agent, cl.user_session.get("agent"))
    chat_model = typing.cast(str, cl.user_session.get("model"))
    memory = typing.cast(Memory, cl.user_session.get("memory", default=get_memory()))

    user_id = "123"
    cb = cl.AsyncLangchainCallbackHandler()

    config = RunnableConfig(
        configurable=dict(
            thread_id=message.thread_id,
            user_id=user_id,
            memory_store=memory,
            model=chat_model,
        ),
        callbacks=[cb],
    )
    response = cl.Message(content="")
    async for event in agent.stream(message.content, config):
        # Send a response back to the user
        await response.stream_token(event)
    await response.send()
