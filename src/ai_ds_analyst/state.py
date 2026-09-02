from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class State(TypedDict):
    user_query: str
    dataset_path: str
    analysis_result: str
    final_answer: str

    messages: Annotated[list, add_messages]