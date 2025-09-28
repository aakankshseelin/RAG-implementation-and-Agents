import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage


os.environ["OPENAI_API_KEY"] = "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"  # Replace with actual or use dotenv


def chat_node(state: dict):
    user_input = state["user_input"]
    history = state.get("history", [])
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    messages = history + [HumanMessage(content=user_input)]
    response = llm.invoke(messages)

    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response.content))

    return {
        "history": history,
        "reply": response.content
    }


graph = StateGraph(dict)
graph.add_node("chat", chat_node)
graph.set_entry_point("chat")
graph.add_edge("chat", END)
graph = graph.compile()

def main():
    st.title(" Chatbot ")

    if "state" not in st.session_state:
        st.session_state.state = {"history": []}

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("You:", key="user_input")
        submitted = st.form_submit_button("Send")

    if submitted and user_input:
        
        full_state = dict(st.session_state.state)
        full_state["user_input"] = user_input

        result = graph.invoke(full_state)
        st.session_state.state["history"] = result["history"]
        st.write(f"**Bot:** {result['reply']}")

    for msg in st.session_state.state["history"]:
        if isinstance(msg, HumanMessage):
            st.markdown(f"**You:** {msg.content}")
        elif isinstance(msg, AIMessage):
            st.markdown(f"**Bot:** {msg.content}")


if __name__ == "__main__":
    main()
