from __future__ import annotations

import json

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.chat_agent_executor import AgentState
from pydantic import BaseModel, Field, create_model

from app import prompts


class State(AgentState):
    today: str


class ToolFormModel(BaseModel):

    name: str
    description: str
    fields: list[str] = Field(default_factory=list)

    def to_llm(self) -> str:
        """Serialize form configuration to LLM input."""
        return json.dumps(
            {
                "instructions": prompts.FORM_INSTRUCTIONS + self.description,
                "required_fields": self.fields,
            }
        )


class SubmitFormInput(BaseModel):
    """Submit form input schema."""

    form_name: str = Field(description="Form name used to collect the information")
    form_data: str = Field(description="The form data collected from the user")
    config: RunnableConfig = Field(description="The configuration of the langgraph")


class BaseFormParser(BaseModel):
    user_id: str
    form_name: str

    @classmethod
    def from_config(cls, form: ToolFormModel) -> PydanticOutputParser[BaseFormParser]:
        """Load form parser from bot model config."""
        form_fields = {k: (str, Field(...)) for k in form.fields}
        form_model = create_model("FormParser", **form_fields, __base__=BaseFormParser)  # type: ignore[call-overload]
        return PydanticOutputParser(pydantic_object=form_model)
