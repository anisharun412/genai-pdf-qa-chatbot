## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex
## AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

## PROBLEM STATEMENT:
The challenge is to develop an agent that can efficiently retrieve and synthesize information from a large corpus of documents, ensuring that it answers queries with precision and relevance, leveraging LlamaIndex for effective retrieval and summarization.

## DESIGN STEPS:
### STEP 1: Data Collection and Preprocessing
- Gather a set of research articles or documents relevant to the topic.
- Preprocess the data by converting the articles into a suitable format (e.g., plain text or structured format).
- Tokenize the content and remove any irrelevant or noisy information.

### STEP 2: Index Construction with LlamaIndex
- Use LlamaIndex (formerly known as GPT Index) to create an index for the documents.
- LlamaIndex will help build an optimized index for efficient retrieval, making it easy to query multiple documents at once.
- Incorporate features like semantic search to improve relevance and accuracy of retrieval.

### STEP 3: Query Handling and Response Generation
- Develop the query interface where users can input questions related to the research articles.
- Integrate the query interface with the LlamaIndex-powered retrieval system.
- Process the retrieved documents to extract relevant information and synthesize a concise response, potentially using additional techniques like summarization.

### STEP 4: Evaluation and Testing
- Test the system with a range of diverse queries to evaluate its performance in terms of accuracy, relevance, and conciseness of responses.
- Collect feedback and refine the system based on test results.

## PROGRAM:

```
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()
import nest_asyncio
nest_asyncio.apply()
urls = [
    "https://openreview.net/forum?id=UYneFzXSJWh",
    "https://openreview.net/forum?id=l4IHywGq6a",
    "https://openreview.net/pdf?id=w1UbdvWH_R3",
]

papers = [
    "fine_tuning_can_distort_pretra.pdf",
    "data_efficient_graph_grammar_l.pdf",
    "neural_collapse_under_mse_loss.pdf",
]

from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]

from llama_index.llms.openai import OpenAI
llm = OpenAI(model="gpt-3.5-turbo")

len(initial_tools)

from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools, 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)

response = agent.query(
    "Tell me about the OOD comparison in fine_tuning_can_distort_pretra, "
    "and then tell me about the evaluation results"
)

response = agent.query("Give me a summary of three files")
print(str(response))
```
## OUTPUT:

<img width="1328" height="691" alt="505102214-e3ccb548-56f6-4d4e-9057-ebf2f98b9978" src="https://github.com/user-attachments/assets/63086832-15db-47b3-9eeb-bec509348a66" />
<img width="1310" height="726" alt="505102854-a9ad8b0c-d7f7-4560-8f8a-19fc94e89019" src="https://github.com/user-attachments/assets/f24d31b2-a5bb-4cab-86f2-97d2ef932d82" />
<img width="1320" height="755" alt="505102943-83c3ce85-0d86-4918-b9f3-4141afb29c4a" src="https://github.com/user-attachments/assets/ec1ee27c-6c04-4330-914f-6d7377a19e61" />

## RESULT:
The system successfully retrieves and synthesizes relevant information from multiple documents, providing concise and relevant answers to the user's query. Performance is evaluated based on the accuracy, relevance, and coherence of the responses.
The system successfully retrieves and synthesizes relevant information from multiple documents, providing concise and relevant answers to the user's query. Performance is evaluated based on the accuracy, relevance, and coherence of the responses.
