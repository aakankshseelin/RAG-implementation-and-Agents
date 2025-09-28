import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.chat_message_histories import SQLChatMessageHistory
import os
from dotenv import load_dotenv
from datetime import datetime
import mysql.connector


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


MYSQL_HOST = "localhost"
MYSQL_USER = "root"
MYSQL_PASSWORD = "12345"
MYSQL_DB = "chatbot_db"


def get_or_create_user(username):
    conn = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB
    )
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
    result = cursor.fetchone()
    if not result:
        cursor.execute("INSERT INTO users (username) VALUES (%s)", (username,))
        conn.commit()
    cursor.close()
    conn.close()


def prompt_user_identity():
    if 'username' not in st.session_state:
        username = st.text_input("Enter your name to continue:", key="username_input")
        if username:
            username = username.strip()
            st.session_state.username = username
            get_or_create_user(username)
            st.session_state.session_id = f"{username.lower()}_session"
            st.session_state.chat_history = get_chat_history(st.session_state.session_id)
            st.success(f"Welcome back, {username}!")
            st.rerun()
        else:
            st.stop()

def get_chat_history(session_id):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string=f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DB}"
    )

def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = get_chat_history(st.session_state.session_id)
    
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
            openai_api_key=OPENAI_API_KEY
        )

    if 'tools' not in st.session_state:
        search = DuckDuckGoSearchRun()
        st.session_state.tools = [
            Tool(
                name="web_search",
                description="Search the internet for current information. Use this for questions about current events, facts, or anything that needs up-to-date information.",
                func=search.run
            )
        ]

    if 'agent_executor' not in st.session_state:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""
                You are a helpful AI assistant with web search capability.
                The user's name is {st.session_state.get('username', 'User')}.
                Address them personally. You can use the web_search tool for current info.
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("assistant", "{agent_scratchpad}")
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
            verbose=True
        )

    if 'processed_messages' not in st.session_state:
        st.session_state.processed_messages = set()

def main():
    st.set_page_config(page_title="Chatbot", page_icon="🤖")
    st.title("🤖 Chatbot")
    st.caption("Ask me anything!")


    prompt_user_identity()
    initialize_session_state()

    if st.button("New Conversation", type="secondary"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.session_id = f"{st.session_state.username.lower()}_{timestamp}"
        st.session_state.chat_history = get_chat_history(st.session_state.session_id)
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
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.agent_executor.invoke({"input": user_input})
                response_text = response["output"]
                if "web_search" in str(response.get("intermediate_steps", [])):
                    response_text = "🔎 *Web Search Result*\n\n" + response_text
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.session_state.chat_history.add_user_message(user_input)
                st.session_state.chat_history.add_ai_message(response_text)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.session_state.messages.pop()
        st.rerun()

if __name__ == "__main__":
    main()
