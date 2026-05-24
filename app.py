import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler # අලුතෙන් ගෙනාපු එක
import os

st.set_page_config(page_title="UniBuddy AI", layout="centered")
st.title("🎓 UniBuddy AI - Smart Assistant")
st.write("Answers from local handbook & live university website.")

db_dir = "./db_storage"
UNI_DOMAIN = "https://www.vau.ac.lk/" 

if not os.path.exists(db_dir):
    st.error("❌ Database not found! Please check deployment.")
else:
    @st.cache_resource
    def load_base_models():
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory=db_dir, embedding_function=embeddings)
        
        try:
            groq_key = st.secrets["GROQ_API_KEY"]
        except:
            st.error("Please add GROQ_API_KEY to Streamlit Secrets!")
            st.stop()
            
        # මෙතන streaming=True කියලා දැම්මා
        llm = ChatGroq(groq_api_key=groq_key, model_name="llama-3.1-8b-instant", streaming=True)
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

    custom_prefix = """You are a helpful assistant for UniBuddy AI. 
The user might ask questions in English, Sinhala, or Singlish (Sinhala language written using Roman/English alphabet). 
You must carefully understand Singlish and Sinhala queries, search the provided documents or tools, and respond accurately in the same language or script the user used.

CRITICAL RULES:
1. Keep your answers VERY concise, short, and to the point.
2. DO NOT provide extra information that the user did not explicitly ask for.
3. Only answer the exact question asked.

Answer the following questions as best you can. You have access to the following tools:"""

    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True, 
        handle_parsing_errors=True,
        agent_kwargs={
            "prefix": custom_prefix
        }
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Ask anything..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # අලුතෙන් දාපු Streaming Callback එක මෙතන තියෙනවා
            st_callback = StreamlitCallbackHandler(st.container())
            
            # Agent දුවද්දී Callback එක පාස් කරනවා
            response = agent.run(user_input, callbacks=[st_callback])
            
            st.markdown(response)
                
        st.session_state.messages.append({"role": "assistant", "content": response})