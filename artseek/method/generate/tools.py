"""The retrieval tool exposed to the MLLM."""

from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from transformers.utils import get_json_schema


@tool(response_format="content_and_artifact")
def get_relevant_documents(
    query: str, requires_image: bool, state: Annotated[dict, InjectedState]
):
    """This is an tool for retrieving relevant documents that can help answer user questions.
    It takes a text query and, optionally, the image to which it refers to retrieve useful contextual knowledge.
    Upon calling the tool, it returns relevant documents that can be used to generate an accurate response.
    Use this tool whenever possible to improve the quality of your answers.

    Args:
        query: A text query for information retrieval.
        requires_image: Whether the query should be combined with the input image or is a text-only query.
        state: The graph state.

    Returns:
        list: A list of retrieved documents relevant to the query.
    """
    # Imported lazily to avoid a circular import (config -> rag -> tools).
    from .config import get_model

    return get_model()._call_retrieve_tool(query, requires_image, state["input_image"])


def get_json_schema_no_state(f: callable) -> dict:
    """JSON schema of a tool function, without the injected `state` argument."""
    schema = get_json_schema(f)
    del schema["function"]["parameters"]["properties"]["state"]
    schema["function"]["parameters"]["required"] = [
        item for item in schema["function"]["parameters"]["required"] if item != "state"
    ]
    return schema


# The schema the model sees (the `state` argument is filled in by the pipeline).
RETRIEVAL_TOOL_SCHEMA = get_json_schema_no_state(get_relevant_documents.func)
