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

#### Step 1: Indexing (The Data Preparation Phase) 

This stage happens **once** to build the knowledge base.

* **Action:** Your code loads the PDF, splits it into small, overlapping text **chunks**, and then uses an **Embedding Model** to convert these chunks into numerical vectors.
* **Purpose:** To make the document searchable by *meaning* rather than just keywords. The **Vector Store (Chroma DB)** is the resulting searchable index.

#### Step 2: Retrieval and Prompt Augmentation (The Search Phase) 

This stage happens **every time** the user asks a question.

* **Action:** The user's question is also converted into a vector. The **Retriever** uses this vector to quickly find the top 7 chunks (documents) from the **Vector Store** that are most semantically similar to the question.
* **Purpose:** To find only the most **relevant facts** from the large document and insert them directly into the LLM's prompt as **context**.


#### Step 3: Generation (The Answering Phase)

This stage uses the language model to create the final answer.

* **Action:** The **ChatGroq LLM** receives the original question *plus* the retrieved context chunks. The **LCEL Chain** directs the model to answer the question *only* using the supplied context.
* **Purpose:** To generate an accurate, conversational response **grounded** in the document's facts, minimizing the risk of the LLM "hallucinating" or making up an answer.

### PROGRAM:
```
Name: Arunsamy D
Reg.No: 212224240016
```
```py
import os
from dotenv import load_dotenv
from pathlib import Path
import shutil
import time # Import time for the sleep function
import uuid # Import uuid for unique directory names

# --- LangChain Core Imports (LCEL/Runnables) ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- Integration Imports (Where objects are instantiated) ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# --- 1. SETUP AND CONFIGURATION ---

# Load environment variables
load_dotenv()

PDF_PATH = "/content/metagpt.pdf"
GROQ_MODEL = "llama-3.1-8b-instant"

# Retrieve GROQ_API_KEY from environment (Colab secrets or .env file)
groq_api_key = os.environ.get('GROQ_API_KEY')

if not groq_api_key:
    print("FATAL: GROQ_API_KEY environment variable not found or is empty.")
    print("Please ensure your GROQ_API_KEY is set in Colab secrets and linked to this notebook, or in a .env file.")
    raise ValueError("GROQ_API_KEY is not set.") # Raise an error to stop execution clearly

if not Path(PDF_PATH).exists():
    print(f"FATAL: Document not found at '{PDF_PATH}'.")
    print("Please place your PDF in the same directory and rename it to 'input_document.pdf'.")
    raise FileNotFoundError(f"Document not found at '{PDF_PATH}'.")

# --- 2. DATA PROCESSING (INDEXING) ---

print(f"--- 🚀 Starting RAG Pipeline for {PDF_PATH} ---")

# Generate a unique directory name for chroma_db to avoid persistent locks
CHROMA_DB_PATH = f"./chroma_db_{uuid.uuid4().hex}"
print(f"Using unique Chroma DB path: {CHROMA_DB_PATH}")

# Remove existing unique chroma_db directory if it somehow exists (unlikely, but good practice)
if os.path.exists(CHROMA_DB_PATH):
    print(f"Clearing existing Chroma DB at {CHROMA_DB_PATH}...")
    shutil.rmtree(CHROMA_DB_PATH)
    time.sleep(1) # Small delay for safety

# 2.1. Load PDF Document
print("1. Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# 2.2. Split Text into Chunks
print("2. Splitting documents into chunks (size=800, overlap=100)...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # Increased from 500
    chunk_overlap=100 # Increased from 50 (ensures smoother overlap)
)
texts = text_splitter.split_documents(documents)

# 2.3. Initialize Embedding Model
print("3. Generating embeddings using 'BAAI/bge-base-en-v1.5'...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# 2.4. Create Vector Store and Retriever
print("4. Storing vectors in Chroma DB and creating retriever...")
vectorstore = Chroma.from_documents(texts, embeddings, persist_directory=CHROMA_DB_PATH)
retriever = vectorstore.as_retriever(search_kwargs={"k": 7})

# --- 3. LCEL RAG CHAIN SETUP ---


# 3.1. Initialize Groq LLM
print(f"5. Initializing Groq LLM: {GROQ_MODEL} (Temperature=0)...")
llm = ChatGroq(
    temperature=0,
    model_name=GROQ_MODEL,
    api_key=groq_api_key # Explicitly pass the key
)

# 3.2. Define Prompt Template (The instruction for the LLM)
prompt = ChatPromptTemplate.from_template("""
You are a highly accurate chatbot. Use ONLY the provided context to answer the user's question.
If the information is not in the context, you MUST state that you cannot find the answer in the document.

Context: {context}
Question: {input}
""")

# 3.3. Create the Document Combination Chain (The 'Stuff' chain)
document_chain = create_stuff_documents_chain(llm, prompt)

# 3.4. Final RAG Chain (using LCEL | operator)
qa_chain = (
    {"context": retriever, "input": RunnablePassthrough()}
    | document_chain
    | StrOutputParser() # The output parser converts the LLM's response into a string
)

# --- 4. INTERACTION LOOP (THE CHATBOT) ---

print("\n--- ✅ Chatbot Ready! Start Asking Questions about your PDF ---")
print("   Type 'exit' to quit the application.")
print("-" * 50)

while True:
    try:
        query = input("You: ")
        if query.lower() == "exit":
            print("Chatbot: Bye! 👋")
            break

        print("\nChatbot: Thinking...")

        answer = qa_chain.invoke(query)

        print(f"\nChatbot: {answer}\n")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        break
```
### OUTPUT:

<img width="1914" height="371" alt="image" src="https://github.com/user-attachments/assets/26f62793-f31c-43ff-8c4d-2e57007d29ad" />

### RESULT:
Thus, a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain was implemented and evaluated for its effectiveness by testing its responses to diverse queries derived from the document's content successfully.
