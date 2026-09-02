import pandas as pd

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

from ai_ds_analyst.state import State

load_dotenv()

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

def analyze_query(state: State):
    print("Analyzing user query with LLM...")

    query = state["user_query"]

    prompt = f"""
You are a data analysis intent classifier.

Classify the user's question into exactly one of these categories:

- revenue
- quantity

User question:
{query}

Return only one word:
revenue
or
quantity
"""

    response = llm.invoke(prompt)

    intent = response.content.strip().lower()

    return {
        "intent": intent
    }

def quantity_analysis(state: State):
    print("Performing quantity analysis...")

    df = pd.read_csv(state["dataset_path"])

    product_quantity = df.groupby("Product")["Quantity"].sum()

    highest_product = product_quantity.idxmax()
    highest_quantity = product_quantity.max()

    result = (
        f"Highest quantity sold: "
        f"{highest_product} ({highest_quantity} units)."
    )

    return {
        "analysis_result": result
    }


def revenue_analysis(state: State):
    print("Performing revenue analysis...")

    df = pd.read_csv(state["dataset_path"])

    df["Revenue"] = df["Quantity"] * df["Price"]

    product_revenue = df.groupby("Product")["Revenue"].sum()

    highest_product = product_revenue.idxmax()
    highest_revenue = product_revenue.max()

    result = (
        f"Highest revenue: "
        f"{highest_product} (₹{highest_revenue:,.0f})."
    )

    return {
        "analysis_result": result
    }

def final_answer(state: State):
    print("Generating final answer with LLM...")

    query = state["user_query"]
    analysis_result = state["analysis_result"]

    prompt = f"""
You are a helpful data analyst.

The user asked:
{query}

The data analysis produced this result:
{analysis_result}

Give the user a clear and concise answer.

Do not perform any new calculations.
Use only the analysis result provided.
"""

    response = llm.invoke(prompt)

    return {
        "final_answer": response.content.strip()
    }    

def route_analysis(state: State):
    intent = state["intent"]

    if intent == "revenue":
        return "revenue_analysis"

    elif intent == "quantity":
        return "quantity_analysis"

    else:
        raise ValueError(
            f"Invalid intent returned by LLM: {intent}"
        )


builder = StateGraph(State)

builder.add_node("analyze_query", analyze_query)
builder.add_node("quantity_analysis", quantity_analysis)
builder.add_node("revenue_analysis", revenue_analysis)
builder.add_node("final_answer", final_answer)
builder.add_edge(START, "analyze_query")
builder.add_conditional_edges(
    "analyze_query",
    route_analysis,
    {
        "quantity_analysis": "quantity_analysis",
        "revenue_analysis": "revenue_analysis",
    }
)
builder.add_edge("quantity_analysis", "final_answer")
builder.add_edge("revenue_analysis", "final_answer")

builder.add_edge("final_answer", END)
graph = builder.compile()




if __name__ == "__main__":
    result = graph.invoke({
        "user_query": "Which product sold the most units?",
        "dataset_path": "data/sales.csv",
        "intent": "",
        "analysis_result": "",
        "final_answer": ""
    })

    print(result)