from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

from ai_ds_analyst.state import State
from ai_ds_analyst.tools import (
    calculate_revenue,
    calculate_quantity,
)


llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

tools = [
    calculate_revenue,
    calculate_quantity,
]

llm_with_tools = llm.bind_tools(tools)


def agent(state: State):
    print("Agent deciding which tool to use...")

    query = state["user_query"]
    dataset_path = state["dataset_path"]

    prompt = f"""
You are a data analyst.

The dataset is located at:
{dataset_path}

User question:
{query}

Choose the appropriate analysis tool.
"""

    response = llm_with_tools.invoke(prompt)

    return {
        "tool_call": response.tool_calls[0]
    }


def execute_tool(state: State):
    print("Executing selected tool...")

    tool_call = state["tool_call"]

    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    if tool_name == "calculate_revenue":
        result = calculate_revenue(**tool_args)

    elif tool_name == "calculate_quantity":
        result = calculate_quantity(**tool_args)

    else:
        raise ValueError(f"Unknown tool: {tool_name}")

    return {
        "analysis_result": result
    }


def final_answer(state: State):
    print("Generating final answer...")

    query = state["user_query"]
    result = state["analysis_result"]

    prompt = f"""
You are a helpful data analyst.

User question:
{query}

Analysis result:
{result}

Give the user a clear and concise answer.

Do not perform any new calculations.
Use only the analysis result provided.
"""

    response = llm.invoke(prompt)

    return {
        "final_answer": response.content.strip()
    }


builder = StateGraph(State)

builder.add_node("agent", agent)
builder.add_node("execute_tool", execute_tool)
builder.add_node("final_answer", final_answer)

builder.add_edge(START, "agent")
builder.add_edge("agent", "execute_tool")
builder.add_edge("execute_tool", "final_answer")
builder.add_edge("final_answer", END)

graph = builder.compile()


if __name__ == "__main__":

    result = graph.invoke({
        "user_query": "Which product sold the most units?",
        "dataset_path": "data/sales.csv",
        "intent": "",
        "analysis_result": "",
        "final_answer": "",
        "tool_call": {},
    })

    print("\nFinal result:")
    print(result)