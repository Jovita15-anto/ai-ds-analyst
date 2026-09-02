from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)

from ai_ds_analyst.state import State
from ai_ds_analyst.tools import (
    calculate_total_revenue,
    analyze_column,
    product_analysis,
    dataset_summary,
    group_analysis,
    dataset_insights,
)


llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

tools = [
    calculate_total_revenue,
    analyze_column,
    product_analysis,
    dataset_summary,
    group_analysis,
    dataset_insights,
]

llm_with_tools = llm.bind_tools(tools)

tool_node = ToolNode(tools)


def agent(state: State):
    print("Agent deciding which tool to use...")

    query = state["user_query"]
    dataset_path = state["dataset_path"]
    messages = state["messages"]

    system_prompt = f"""You are a data analyst working with a dataset.

Dataset path: {dataset_path}

Rules:

- Use calculate_total_revenue for total revenue questions.
- Use analyze_column for calculations on one numeric column.
- Use product_analysis for highest/lowest product analysis.
- Use dataset_summary for dataset overview.
- Use group_analysis for calculations broken down by another column.
- Use dataset_insights when the user asks for key insights,
  important findings, business insights, or overall dataset insights.

IMPORTANT GROUPING RULE:

If the user asks for a metric "by", "per", "for each", "broken down by",
or asks for a metric separately for each category/group, ALWAYS use
group_analysis.

Examples:

"average salary by department"
→ group_analysis
→ group_by: department
→ metric: salary
→ operation: average

"total sales by city"
→ group_analysis
→ group_by: city
→ metric: sales
→ operation: sum

"average orders per city"
→ group_analysis
→ group_by: city
→ metric: orders
→ operation: average

"highest salary"
→ analyze_column
→ metric: salary
→ operation: highest

"average salary"
→ analyze_column
→ metric: salary
→ operation: average

ERROR RECOVERY RULE:

If a previous tool result says that a column was not found:

1. Do NOT guess the available columns.
2. Call dataset_summary to inspect the dataset.
3. Use the actual columns returned by dataset_summary.
4. Do not perform the original calculation if the requested
   column does not exist.
5. After dataset_summary gives the available columns,
   provide a clear explanation to the user.

IMPORTANT:

- Only call a tool if it directly answers an explicit part of the user's question.
- Never perform additional or unrelated analysis.
- For multiple parts, call exactly the tools needed for those parts.
- Do not call calculate_total_revenue unless the user explicitly asks for total revenue.
- Do not call product_analysis unless the user explicitly asks about highest/lowest products.
- Do not call dataset_insights unless the user explicitly asks for insights.
- Do not generate Python code.
- After all requested parts are answered, return a concise natural-language answer.

MULTI-PART QUESTION HANDLING:

When the user asks for more than one analysis, break the question
into separate tasks before selecting tools.

For each task:
1. Identify the exact metric.
2. Identify the exact operation.
3. Identify whether grouping is required.
4. Select the tool that matches that task.

Example:

Question:
"What is the total revenue by category and which product has the highest price?"

This contains TWO separate tasks:

Task 1:
- Analysis: total revenue by category
- Tool: group_analysis
- group_by: category
- metric: revenue
- operation: sum

Task 2:
- Analysis: product with highest price
- Tool: product_analysis
- metric: price
- operation: highest

You MUST complete both tasks before producing the final answer.

Do not replace "revenue" with "price".
Do not replace "highest price" with "maximum price by category".
Do not stop after completing only one task.

After the first successful tool result, check the original
user question again and determine whether another task remains.
If another task remains, call the appropriate tool.
"""

    if not messages:
        messages = [HumanMessage(content=query)]

    full_messages = [SystemMessage(content=system_prompt)] + messages

    response = llm_with_tools.invoke(full_messages)

    # Force-correct dataset_path in every tool call —
    # never trust the model to reproduce the exact path string.
    for tool_call in response.tool_calls:
        if "dataset_path" in tool_call["args"]:
            tool_call["args"]["dataset_path"] = dataset_path

    print("AI RESPONSE:", response)
    print("TOOL CALLS:", response.tool_calls)

    return {
        "messages": [response]
    }

def should_continue(state: State):
    last_message = state["messages"][-1]

    if isinstance(last_message, ToolMessage):

        content = last_message.content or ""

        is_error = (
            getattr(last_message, "status", None) == "error"
            or "not found" in content.lower()
            or "unsupported" in content.lower()
            or content.startswith("Error invoking tool")
        )

        if is_error:
            return "agent"

        # Successful tool result:
        # give the agent another chance to check
        # whether the entire question is answered.
        return "agent"

    if last_message.tool_calls:
        return "tools"

    return "final_answer"

def route_after_tools(state: State):
    last_message = state["messages"][-1]

    if isinstance(last_message, ToolMessage):

        content = last_message.content or ""

        is_error = (
            getattr(last_message, "status", None) == "error"
            or content.startswith("Error invoking tool")
            or "not found" in content.lower()
            or "unsupported" in content.lower()
        )

        if is_error:
            return "recover"

        return "final_answer"

    return "agent"

def final_answer(state: State):
    print("Generating final answer...")

    messages = state["messages"]

    # Check the latest AI response first.
    # This is especially important after error recovery.
   

    # Collect successful tool results
    tool_results = []

    for msg in messages:

        if isinstance(msg, ToolMessage):

            content = msg.content or ""

            is_error = (
                getattr(msg, "status", None) == "error"
                or content.startswith("Error invoking tool")
                or "not found" in content.lower()
                or "unsupported" in content.lower()
            )

            if not is_error and content.strip():
                tool_results.append(content)

    if tool_results:

        tool_result = "\n\n".join(tool_results)

        query = state["user_query"]
        dataset_path = state["dataset_path"]

        prompt = f"""
You are a professional data analyst.

Dataset: {dataset_path}
User question: {query}

Analysis result:
{tool_result}

Answer the user's question directly.

Rules:
- Use only the information provided in the analysis result.
- Do not invent numbers.
- Do not perform additional analysis.
- Do not suggest unrelated analyses.
- When the analysis result contains a person, employee, customer, or product name, you MUST include that exact name in the answer.
- Never replace a name from the analysis result with "x", "unknown", or any invented name.
- Preserve names and values exactly as provided by the analysis result.
- Keep the answer concise and professional.

FORMAT THE ANSWER EXACTLY AS FOLLOWS:

- Start with a short plain-text heading.
- Put the heading on its own line.
- Then put each answer as a separate bullet point.
- Each bullet must be on its own line.
- Use "-" for bullets.
- NEVER use "•" bullets.
- NEVER combine multiple bullet points into one paragraph.
- NEVER use markdown tables.
- NEVER use bold text, asterisks, or markdown formatting.
- Do not repeat the same fact.
- For grouped results, put each group on a separate bullet.
- Keep the answer concise.

Example:

Average Salary by Department

- HR: ₹50,000
- IT: ₹85,000
- Sales: ₹71,000
"""

        response = llm.invoke(prompt)

        return {
            "final_answer": response.content.strip(),
            "analysis_result": tool_result
        }

    return {
        "final_answer": "The analysis could not be completed.",
        "analysis_result": ""
    }

def recover_from_error(state: State):
    print("Tool failed. Checking dataset structure...")

    dataset_path = state["dataset_path"]

    result = dataset_summary.invoke({
        "dataset_path": dataset_path
    })

    return {
        "messages": [
            ToolMessage(
                content=result,
                name="dataset_summary",
                tool_call_id="recovery"
            )
        ]
    }

builder = StateGraph(State)

builder.add_node("agent", agent)
builder.add_node("tools", tool_node)
builder.add_node("recover", recover_from_error)
builder.add_node("final_answer", final_answer)

builder.add_edge(START, "agent")

builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "agent": "agent",
        "final_answer": "final_answer",
    }
)

builder.add_conditional_edges(
    "tools",
    route_after_tools,
    {
        "recover": "recover",
        "final_answer": "final_answer",
        "agent": "agent",
    }
)

builder.add_edge("recover", "agent")

builder.add_edge("final_answer", END)

graph = builder.compile()

if __name__ == "__main__":

    result = graph.invoke(
    {
        "user_query": "What are the key insights from this dataset?",
        "dataset_path": "data/sales.csv",
        "analysis_result": "",
        "final_answer": "",
        "messages": [],
    },
)

    print("\nFinal result:")
    print(result)