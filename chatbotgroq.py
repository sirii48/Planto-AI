import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit app configuration
st.set_page_config(page_title="Grok Chatbot", page_icon="🤖")
st.title("Grok-Powered Chatbot")
st.markdown("Ask me anything, and I'll respond with the power of Groq and LangChain!")

# Sidebar for customization
st.sidebar.title("Chatbot Settings")
system_prompt = st.sidebar.text_input("System Prompt:", value="You are a helpful assistant.")
model = st.sidebar.selectbox("Choose a Model", ["llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma-7b-it"])
memory_length = st.sidebar.slider("Conversational Memory Length:", 1, 10, value=5)

# Initialize conversational memory
memory = ConversationBufferWindowMemory(k=memory_length, memory_key="chat_history", return_messages=True)

# Save chat history to memory
for message in st.session_state.chat_history:
    memory.save_context({"input": message["human"]}, {"output": message["AI"]})

# Initialize Groq LangChain chat object
groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model, temperature=0)

# Construct the chat prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{human_input}")
])

# Create the conversation chain
conversation = LLMChain(
    llm=groq_chat,
    prompt=prompt,
    memory=memory,
    verbose=False
)

# User input
user_question = st.text_input("Your Question:", placeholder="Type your question here...")

# Process user input and generate response
if user_question:
    with st.spinner("Generating response..."):
        response = conversation.predict(human_input=user_question)
        message = {"human": user_question, "AI": response}
        st.session_state.chat_history.append(message)
        st.write("**You:** ", user_question)
        st.write("**Chatbot:** ", response)

# Display chat history
if st.session_state.chat_history:
    st.subheader("Conversation History")
    for msg in st.session_state.chat_history:
        st.write(f"**You:** {msg['human']}")
        st.write(f"**Chatbot:** {msg['AI']}")
        st.markdown("---")