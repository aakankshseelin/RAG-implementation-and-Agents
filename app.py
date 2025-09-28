import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from datetime import datetime


load_dotenv()


def get_chat_history():
    history_file = "chat_history.json"
    if not os.path.exists(history_file):
        with open(history_file, 'w') as f:
            f.write('[]')
    return FileChatMessageHistory(history_file)


def load_vectorstore():
    try:
        embeddings = OpenAIEmbeddings()
        if os.path.exists("faiss_index"):
            return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        else:
            st.warning("No FAISS index found. Please run rag_setup.py first.")
            return None
    except Exception as e:
        st.error(f"Error loading vectorstore: {str(e)}")
        return None


def rag_search_tool(query: str) -> str:
    try:
        if 'vectorstore' not in st.session_state or st.session_state.vectorstore is None:
            return "RAG vectorstore not available. Please ensure FAISS index is loaded."

        docs = st.session_state.vectorstore.similarity_search(query, k=3)

        if not docs:
            return "No relevant documents found."

        context = "\n\n".join([f"Document {i+1}: {doc.page_content}" for i, doc in enumerate(docs)])
        return f"Retrieved context:\n{context}"

    except Exception as e:
        return f"Error in RAG search: {str(e)}"


def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = get_chat_history()

    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            chat_memory=st.session_state.chat_history,
            memory_key="chat_history",
            return_messages=True
        )

    if 'llm' not in st.session_state:
        st.session_state.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = load_vectorstore()

    if 'tools' not in st.session_state:
        tavily_tool = TavilySearchResults(k=3)
        st.session_state.tools = [
            Tool(
                name="web_search",
                description="Search the internet for current information.",
                func=tavily_tool.run
            ),
            Tool(
                name="rag_search",
                description="Search the company knowledge base.",
                func=rag_search_tool
            )
        ]

    if 'agent_executor' not in st.session_state:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
            You are a helpful AI assistant for TechNova Solutions with web and internal document search capabilities.

            Tools:
            1. web_search: Use for current events, live data, or general internet queries.
            2. rag_search: Use for company-specific knowledge (TechNova).

            Always try rag_search first for company-related questions.
            Cite sources when needed and be helpful.
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        agent = create_openai_functions_agent(
            llm=st.session_state.llm,
            tools=st.session_state.tools,
            prompt=prompt
        )

        st.session_state.agent_executor = AgentExecutor(
            agent=agent,
            tools=st.session_state.tools,
            memory=st.session_state.memory,
            verbose=True,
            handle_parsing_errors=True
        )

    if 'processed_messages' not in st.session_state:
        st.session_state.processed_messages = set()


def main():
    st.set_page_config(page_title="Chatbot", page_icon="🤖")
    st.title("🤖 Chatbot")
    st.caption("Ask me anything!")

    initialize_session_state()

    if st.session_state.vectorstore is not None:
        st.success("✅ Company knowledge base loaded.")
    else:
        st.error(" RAG vectorstore not available.")

    if st.button("New Conversation", type="secondary"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if os.path.exists("chat_history.json"):
            new_name = f"chat_history_{timestamp}.json"
            counter = 1
            while os.path.exists(new_name):
                new_name = f"chat_history_{timestamp}_{counter}.json"
                counter += 1
            os.rename("chat_history.json", new_name)
        st.session_state.chat_history = get_chat_history()
        st.session_state.messages = []
        st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []
        for message in st.session_state.chat_history.messages:
            st.session_state.messages.append({"role": message.type, "content": message.content})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Ask me anything...")
    if user_input:
        st.session_state.messages.append({"role": "human", "content": user_input})
        with st.chat_message("human"):
            st.write(user_input)

        with st.spinner("Thinking..."):
            try:
                response = st.session_state.agent_executor.invoke({"input": user_input})
                response_text = response["output"]
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.session_state.chat_history.add_user_message(user_input)
                st.session_state.chat_history.add_ai_message(response_text)

                with st.chat_message("assistant"):
                    st.write(response_text)

            except Exception as e:
                st.error(f"Error: {str(e)}")
                if st.session_state.messages and st.session_state.messages[-1]["role"] == "human":
                    st.session_state.messages.pop()

if __name__ == "__main__":
    main()
