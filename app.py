import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
import os

st.set_page_config(page_title="UniBuddy AI", layout="centered")
st.title("🎓 UniBuddy AI - Smart Assistant")
st.write("Answers from local handbook & live university website.")

db_dir = "./db_storage"
UNI_DOMAIN = "uom.lk" # ඔයාගේ යුනිවර්සිටි ඩොමේන් එකට වෙනස් කරගන්න

if not os.path.exists(db_dir):
    st.error("❌ Database not found! Please check deployment.")
else:
    @st.cache_resource
    def load_base_models():
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory=db_dir, embedding_function=embeddings)
        
        # Secrets හරහා API Key එක ගැනීම
        try:
            groq_key = st.secrets["GROQ_API_KEY"]
        except:
            st.error("Please add GROQ_API_KEY to Streamlit Secrets!")
            st.stop()
            
        llm = ChatGroq(groq_api_key=groq_key, model_name="llama3-8b-8192")
        return vectorstore, llm

    vectorstore, llm = load_base_models()

    def search_local_db(query):
        docs = vectorstore.similarity_search(query, k=2)
        return "\n".join([d.page_content for d in docs])

    ddg_search = DuckDuckGoSearchRun()
    def search_university_website(query):
        advanced_query = f"site:{UNI_DOMAIN} {query}"
        return ddg_search.run(advanced_query)

    from langchain.tools import Tool
    tools = [
        Tool(
            name="Local_Knowledge_Base",
            func=search_local_db,
            description="Use this to answer questions about university guidelines, lecturer details, policies, and course content."
        ),
        Tool(
            name="University_Live_Website",
            func=search_university_website,
            description=f"Use this ONLY for finding latest live notices, news, or events from the university website {UNI_DOMAIN}."
        )
    ]

    if "messages" not in st.session_state:
        st.session_state.messages = []

    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True, 
        handle_parsing_errors=True
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Ask anything..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Agent is searching..."):
                response = agent.run(user_input)
                st.markdown(response)
                
        st.session_state.messages.append({"role": "assistant", "content": response})