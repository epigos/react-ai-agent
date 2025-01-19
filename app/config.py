import typing
from dataclasses import dataclass, fields
from pathlib import Path

import pydantic
from langchain_core.runnables import RunnableConfig, ensure_config
from mem0 import Memory
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Settings for this project
    """

    root_dir: Path = Path(__file__).parent.parent

    debug: bool = False
    log_level: str = "INFO"
    log_format: typing.Literal["json", "console"] = "console"
    graphs_dir: Path = root_dir / "assets"
    langchain_tracing_v2: str = "true"
    langchain_api_key: str = ""
    langchain_project: str = "react-agent"
    embeddings_model: str = "openai:text-embedding-3-small"
    retriever_threshold: float = 0.3
    # openai config
    openai_api_key: pydantic.SecretStr = pydantic.SecretStr("")

    model_config = SettingsConfigDict(
        env_file=f"{root_dir}/.env", env_file_encoding="utf-8"
    )


@dataclass(kw_only=True)
class AgentConfiguration:
    thread_id: str
    user_id: str
    model: str
    memory_store: Memory

    @classmethod
    def from_runnable_config(
        cls, config: RunnableConfig | None = None
    ) -> "AgentConfiguration":
        """Create a Configuration instance from a RunnableConfig object."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})


settings = Settings()
