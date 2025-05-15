from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pytesseract
from pdf2image import convert_from_path
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.language_models import LLM
from typing import Optional, List
import requests

# Initialize FastAPI
app = FastAPI()

# Path to tesseract (Update if yours is different)
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
#  Path to your PDF
pdf_path = r"C:/Users/chd73/Downloads/intern/the-psychology-of-money.pdf"

#  Groq LLM setup
class GroqLLM(LLM):
    model: str
    groq_api_key: str

    @property
    def _llm_type(self) -> str:
        return "custom-groq"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
        if response.status_code != 200:
            raise Exception(f"Groq API Error: {response.text}")
        return response.json()["choices"][0]["message"]["content"].strip()

from langchain.schema import Document  # Add this import

# Load and OCR the PDF
print("Converting PDF to images...")
images = convert_from_path(pdf_path, poppler_path=r"C:/poppler/poppler-24.08.0/Library/bin")
docs = []

for i, img in enumerate(images):
    text = pytesseract.image_to_string(img)
    if text.strip():
        docs.append(Document(page_content=text, metadata={"page": i}))  # Use Document objects

print(f"Extracted text from {len(docs)} pages.")

# Chunk the documents
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)
print(f"Number of chunks created: {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk.page_content[:200]}...")  # Print the first 200 characters of each chunk

if len(chunks) == 0:
    print("No chunks created. Please check your PDF and OCR results.")
    exit()

#  Create vector store
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding_model)


GROQ_API_KEY = "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"
llm = GroqLLM(model="mistral-saba-24b", groq_api_key=GROQ_API_KEY)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an AI assistant. Use the following context to answer the question:\n\n"
        "Context: {context}\n\n"
        "Question: {question}\n\n"
        "If the context does not contain relevant information, respond with 'I cannot find the answer in the provided context.'"
    )
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

# Define the input model for the API
class Query(BaseModel):
    question: str

# API endpoint for querying the chatbot
@app.post("/ask")
def ask_question(query: Query):
    try:
        # Retrieve relevant documents
        retrieved_docs = vectorstore.as_retriever().get_relevant_documents(query.question)
        if not retrieved_docs:
            return {"question": query.question, "answer": "I could not find any relevant information in the provided context."}
        
        # Generate the answer
        answer = qa_chain.invoke({"query": query.question})
        return {"question": query.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))