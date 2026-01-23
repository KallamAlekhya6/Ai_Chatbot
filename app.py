import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

# =========================================================
# Prompt Template
# =========================================================

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer clearly and concisely."),
        ("human", "{question}")
    ]
)

# =========================================================
# LLM + Chain
# =========================================================

def generate_response(question, model_name, temperature):
    llm = ChatOllama(
        model=model_name,
        temperature=temperature
    )

    output_parser = StrOutputParser() 
    chain = prompt | llm | output_parser

    return chain.invoke({"question": question})

# =========================================================
# Streamlit UI
# =========================================================

st.set_page_config(page_title="Ollama Q&A Chatbot", page_icon="🦙")

st.title("Alekhya  AI Chatbot ")

st.sidebar.header("Model Settings")

model_name = st.sidebar.selectbox(
    "Select Open Source Model",
    ["mistral", "llama3", "gemma"]
)

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.7
)

st.write("Ask any question 👇")

user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, model_name, temperature)
    st.markdown("### 🤖 Assistant")
    st.write(response)
else:
    st.info("please enter a question")

st.write("\n---\n")
st.write("App by: ** kallam Alekhya **")
