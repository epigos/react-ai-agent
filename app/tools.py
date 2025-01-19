import logging
import typing
from datetime import datetime

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.embeddings_filter import EmbeddingsFilter
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import (
    BaseTool,
    BaseToolkit,
    Tool,
    create_retriever_tool,
    tool,
)
from pydantic import BaseModel, Field

from app import prompts, schemas, utils
from app.config import AgentConfiguration, settings

logger = logging.getLogger(__name__)

# Sample Customer Data (replace with actual database/retrieval method)
customer_data = {
    "123": {"name": "Alice Smith", "email": "alice@example.com", "orders": 5},
    "456": {"name": "Bob Johnson", "email": "bob@example.com", "orders": 2},
}

# sample form data
forms_dict = {
    "address_change": {
        "description": "Please provide your new address.",
        "fields": ["address", "city"],
    },
    "book_appointment": {
        "description": "Please provide your preferred date and time.",
        "fields": ["date", "time"],
    },
    "open_account": {
        "description": "Please provide your account the following information.",
        "fields": ["name", "phone_number", "email", "location"],
    },
}
# sample knowledge base
knowledge_base_docs = [
    Document(
        page_content="Our support team is available 24/7 to assist with your inquiries."
    ),
    Document(
        page_content="We offer a 30-day return policy for unused and unopened items."
    ),
    Document(page_content="Shipping is free for orders above $50."),
    Document(
        page_content="Payment options include credit card, PayPal, and Apple Pay."
    ),
    Document(
        page_content="For technical issues, please contact techsupport@company.com."
    ),
    Document(
        page_content="Gift cards can be purchased online and are delivered via email."
    ),
    Document(
        page_content="To reset your password, click on 'Forgot Password' on the login page."
    ),
    Document(
        page_content="Our physical stores are open from 9 AM to 9 PM, Monday through Saturday."
    ),
    Document(page_content="Shipping takes 2-3 days."),
]


@tool(parse_docstring=True)
def get_customer_info(customer_id: str) -> str:
    """
    Retrieves customer information (name, orders) based on ID.
    You must ask the user to provide the ID if it's not available.

    Args:
        customer_id (str): The customer ID.
    """
    logger.info("Retrieving customer information for %s", customer_id)
    if customer_id in customer_data:
        return str(customer_data[customer_id])
    else:
        raise ValueError("Customer not found.")


class FormTool(BaseTool):  # type: ignore[override]
    """Tool to retrieve form fields."""

    form: schemas.ToolFormModel = Field(exclude=True)

    def _run(self, run_manager: CallbackManagerForToolRun | None = None) -> str:
        """Retrieves form fields for the user to complete."""
        logger.info("Retrieving form for %s", self.form.name)
        return self.form.to_llm()


class SubmitForm(BaseTool):  # type: ignore[override]
    """Tool to submit form data."""

    description: str = "Submits the form data collected from the user."
    args_schema: typing.Type[BaseModel] = schemas.SubmitFormInput
    form_registry: dict[str, schemas.ToolFormModel] = Field(exclude=True)
    llm: BaseChatModel = Field(exclude=True)

    def _run(
        self,
        form_name: str,
        form_data: str,
        config: RunnableConfig,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Submits the form data collected from the user in JSON format.

        Args:
            form_name (str): Form name used to collect the information
            form_data (str): The form data collected from the user.
            config (RunnableConfig): The runnable config.
        """
        cfg = AgentConfiguration.from_runnable_config(config)
        try:
            form = self.form_registry[form_name]
        except KeyError as exc:
            raise ValueError(f"Form `{form_name}` not found.") from exc

        parser = schemas.BaseFormParser.from_config(form)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    prompts.SUBMIT_FORM_PROMPT,
                ),
                ("system", "Today is {today}"),
                ("human", "{query}"),
            ]
        ).partial(
            format_instructions=parser.get_format_instructions(),
            today=datetime.now().isoformat(),
        )
        chain = prompt | self.llm | parser

        query = f"""User ID: {cfg.user_id}.\n
            Form ID: {form.name}.\n
            Form data: {form_data}
        """
        res = chain.invoke({"query": query}, config)

        logger.info("Form submitted: %s", res)
        return "Form successfully submitted. An langgraph will get back to you shortly."


@tool(parse_docstring=True)
def save_memory(context: str, config: RunnableConfig) -> str:
    """
    Store information between conversations to memory to build a comprehensive understanding of the user.

    Args:
        context (str): Context of the conversation to save.
        config (RunnableConfig): The runnable config.
    """
    cfg = AgentConfiguration.from_runnable_config(config)
    res = cfg.memory_store.add(context, user_id=cfg.user_id)
    return f"Memory saved: {res}"


@tool(parse_docstring=True)
def search_memory(query: str, config: RunnableConfig) -> str:
    """
    Search memory for relevant information about the user.

    Args:
        query (str): Query to retrieve from memory.
        config (RunnableConfig): The runnable config.
    """
    cfg = AgentConfiguration.from_runnable_config(config)
    res = cfg.memory_store.search(query, user_id=cfg.user_id, limit=3)

    memories = utils.process_recall_memory(dict(res))
    return ".\n".join(memories)


class AgentToolkit(BaseToolkit):
    """Toolkit for retrieving langgraph tools."""

    llm: BaseChatModel = Field(exclude=True)

    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)

        self._form_registry = {
            key: schemas.ToolFormModel(
                name=key,
                description=str(form["description"]),
                fields=list(form["fields"]),
            )
            for key, form in forms_dict.items()
        }

    def get_tools(self) -> list[BaseTool]:
        retriever_tool = self._get_retriever_tool()
        form_tools = self._get_form_tools()

        return [
            save_memory,
            search_memory,
            retriever_tool,
            get_customer_info,
        ] + form_tools

    def _get_form_tools(self) -> list[BaseTool]:
        form_tools: list[BaseTool] = [
            FormTool(
                name=form.name,
                description=f"Provides form details for {form.name}",
                return_direct=False,
                form=form,
            )
            for form in self._form_registry.values()
        ]

        submit_form = SubmitForm(
            name="submit_form",
            return_direct=False,
            form_registry=self._form_registry,
            llm=self.llm,
        )
        return form_tools + [submit_form]

    @staticmethod
    def _get_retriever_tool() -> Tool:
        embeddings = utils.load_embeddings_model()
        db = FAISS.from_documents(knowledge_base_docs, embeddings)

        embeddings_filter = EmbeddingsFilter(
            embeddings=embeddings, similarity_threshold=settings.retriever_threshold
        )
        retriever = db.as_retriever()
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=embeddings_filter, base_retriever=retriever
        )

        return create_retriever_tool(
            compression_retriever,
            "company_knowledge_base",
            "Search and return all information about the company.",
        )
