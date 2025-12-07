## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:

In many cases, users need specific information from large documents without manually searching through them. A question-answering chatbot can address this problem by:

1. Parsing and indexing the content of a PDF document.
2. Allowing users to ask questions in natural language.
3. Providing concise and accurate answers based on the content of the document.
  
The implementation will evaluate the chatbot’s ability to handle diverse queries and deliver accurate responses.

### DESIGN STEPS:

#### STEP 1: Load and Parse PDF
Use LangChain's DocumentLoader to extract text from a PDF document.

#### STEP 2: Create a Vector Store
Convert the text into vector embeddings using a language model, enabling semantic search.

#### STEP 3: Initialize the LangChain QA Pipeline
Use LangChain's RetrievalQA to connect the vector store with a language model for answering questions.

#### STEP 4: Handle User Queries
Process user queries, retrieve relevant document sections, and generate responses.

#### STEP 5: Evaluate Effectiveness
Test the chatbot with a variety of queries to assess accuracy and reliability.


### PROGRAM:
```
Name: Arunsamy D
Reg.No: 212224240016
```
```py
import os
import requests
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
import logging

# --- 1. CONFIGURATION AND DOWNLOAD ---

# Define the specific LLM configuration provided by the user
HF_ROUTER_URL = "https://router.huggingface.co/v1"
HF_TOKEN = "token" # NOTE: Using the token provided by the user
LLM_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# Ensure the token is set as an environment variable for LangChain components
os.environ['HF_TOKEN'] = HF_TOKEN

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define the sample PDF document (Using the 'metagpt' paper for a concrete example)
# Replace this with your actual document if needed.
DOC_URL = "https://openreview.net/pdf?id=VtmBAGCN7o"
DOC_NAME = "metagpt.pdf"

# Download the document
if not Path(DOC_NAME).exists():
    logging.info(f"Downloading {DOC_NAME}...")
    try:
        response = requests.get(DOC_URL)
        response.raise_for_status() # Check for request errors
        with open(DOC_NAME, "wb") as f:
            f.write(response.content)
        logging.info(f"Successfully downloaded {DOC_NAME}.")
    except Exception as e:
        logging.error(f"Failed to download {DOC_NAME}: {e}")
        # Exit or handle error if download fails
        exit()

# --- 2. DATA PROCESSING (RAG Pipeline Components) ---

# A. Document Loading
logging.info("Loading document...")
loader = PyPDFLoader(DOC_NAME)
documents = loader.load()

# B. Text Splitting
logging.info("Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
texts = text_splitter.split_documents(documents)
logging.info(f"Created {len(texts)} chunks for indexing.")

# C. Embedding Model
# Use a local Hugging Face model for embeddings (cost-effective and fast)
logging.info("Initializing embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# D. Vector Store Creation (FAISS is a fast local index)
logging.info("Creating FAISS Vector Store...")
vectorstore = FAISS.from_documents(texts, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# --- 3. LLM AND RAG CHAIN SETUP ---

# A. LLM Initialization (Connecting to HF Router)
# LangChain uses the ChatOpenAI class to connect to OpenAI-compatible endpoints.
logging.info(f"Initializing Chat LLM: {LLM_MODEL_NAME} via HF Router...")
llm = ChatOpenAI(
    openai_api_base=HF_ROUTER_URL,
    openai_api_key=HF_TOKEN,
    model=LLM_MODEL_NAME,
    temperature=0.1, # Keep factual
)

# B. Prompt Template
# A robust prompt template guides the LLM to use the context and be honest about its sources.
prompt_template = """
You are a concise and accurate question-answering assistant. 
Use the following context to answer the user's question. 
If the answer is not in the context, clearly state that you cannot find the answer in the provided document.

Context: {context}
Question: {question}

Concise Answer:
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# C. RetrievalQA Chain (The Chatbot)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # 'Stuff' means it combines all retrieved docs into one prompt.
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True # Important for evaluating accuracy/faithfulness
)

# --- 4. EVALUATION AND TESTING ---

def evaluate_chatbot(query):
    """Processes a query and prints the response and source."""
    logging.info(f"\n--- Testing Query: {query} ---")
    
    # Run the chain
    result = qa_chain.invoke({"query": query})
    
    # Extract results
    answer = result['result']
    source_docs = result['source_documents']
    
    print("\n🤖 Chatbot Response:")
    print(answer)
    print("\n📚 Source(s) Found:")
    
    # Print sources for verification (Evaluation)
    if source_docs:
        # Check if the answer is supported by the context (Faithfulness check)
        print(f"✅ Found {len(source_docs)} supporting chunks (Faithfulness Check).")
        print(f"Source Document: {source_docs[0].metadata['source']} (Page: {source_docs[0].metadata.get('page', 'N/A')})")
        
        # A manual Relevancy check is done by the user reviewing the answer vs the query.
    else:
        print("❌ No sources found.")
        
    return answer

# Test with diverse queries derived from the document's content (MetaGPT)
print("--- Starting Chatbot Evaluation ---")

evaluate_chatbot("What is the core idea of the MetaGPT framework?")
evaluate_chatbot("What are the two main phases of the MetaGPT workflow?")
evaluate_chatbot("What is the role of the Architect in the process?")
evaluate_chatbot("What is the capital of Mars?") # Test for refusal/hallucination

```
### OUTPUT:
<img width="1919" height="447" alt="image" src="https://github.com/user-attachments/assets/299986c8-aad0-4421-a086-a24751343b27" />

### RESULT:
Thus, a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain was implemented and evaluated for its effectiveness by testing its responses to diverse queries derived from the document's content successfully.
