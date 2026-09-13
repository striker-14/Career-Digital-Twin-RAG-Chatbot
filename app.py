import gradio as gr

from dotenv import load_dotenv

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from context import SYSTEM_PROMPT
from rag.retriever import get_retriever
from tools import (
    record_user_details,
    record_unknown_question,
)

load_dotenv(override=True)


# -----------------------------
# LLM
# -----------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
)


# -----------------------------
# Tools
# -----------------------------

tools = [
    record_user_details,
    record_unknown_question,
]

tool_map = {
    tool.name: tool
    for tool in tools
}

llm_with_tools = llm.bind_tools(tools)


# -----------------------------
# Retriever
# -----------------------------

retriever = get_retriever()


# -----------------------------
# Prompt
# -----------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("placeholder", "{messages}"),
    ]
)


# -----------------------------
# History
# -----------------------------

def convert_history(history):
    messages = []

    for item in history or []:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content", "")

        if not isinstance(content, str):
            continue

        if role == "user":
            messages.append(
                HumanMessage(content=content)
            )

        elif role == "assistant":
            messages.append(
                AIMessage(content=content)
            )

    return messages


# -----------------------------
# Retrieved documents
# -----------------------------

def format_documents(documents):
    if not documents:
        return (
            "No relevant information was found "
            "in the knowledge base."
        )

    return "\n\n".join(
        f"[Source {i}]\n{doc.page_content}"
        for i, doc in enumerate(documents, start=1)
    )


# -----------------------------
# Chat
# -----------------------------

def chat(message, history):
    history_messages = convert_history(history)

    # RAG: retrieve only relevant information
    documents = retriever.invoke(message)
    context = format_documents(documents)

    messages = history_messages + [
        HumanMessage(content=message)
    ]

    prompt_messages = prompt.invoke(
        {
            "context": context,
            "messages": messages,
        }
    ).to_messages()

    response = llm_with_tools.invoke(prompt_messages)

    # Tool-calling loop
    while response.tool_calls:
        prompt_messages.append(response)

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            print(f"Tool called: {tool_name}")

            selected_tool = tool_map.get(tool_name)

            if selected_tool is None:
                result = f"Unknown tool: {tool_name}"
            else:
                result = selected_tool.invoke(tool_args)

            prompt_messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call_id,
                )
            )

        response = llm_with_tools.invoke(
            prompt_messages
        )

    return response.content


# -----------------------------
# Gradio
# -----------------------------

demo = gr.ChatInterface(
    fn=chat,
    title="My Digital Twin",
    description=(
        "Ask me about my career, skills, "
        "projects and experience."
    ),
)


if __name__ == "__main__":
    demo.launch(inbrowser=True)
