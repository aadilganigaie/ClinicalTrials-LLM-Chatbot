!pip install -q transformers einops accelerate langchain bitsandbytes sentence_transformers faiss-cpu pypdf sentencepiece 
from langchain import HuggingFacePipeline 
from transformers import AutoTokenizer 
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.document_loaders.csv_loader import CSVLoader 
from langchain.vectorstores import FAISS, Chroma
from langchain.chains import RetrievalQA 
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
import accelerate
import transformers 
import torch 
import textwrap 

# Load Data, You can load any type of data
loader = CSVLoader('File path', encoding="utf-8", csv_args={'delimiter': ','}) 
data = loader.load() 

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',model_kwargs={'device': 'cpu'}) 

db = FAISS.from_documents(data, embeddings)

model = "HuggingFaceH4/zephyr-7b-beta" 

tokenizer = AutoTokenizer.from_pretrained(model, token = token) 
pipeline = transformers.pipeline("text-generation", model=model,tokenizer=tokenizer, torch_dtype=torch.bfloat16, trust_remote_code=True, 
device_map="auto", max_length=1000, do_sample=True, top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id ) 

llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0}) 

template = """
### Instruction: You are a Clinical Data Scientist and Data Analyst with expertise in statistical data analysis and report generation. Your focus is on delivering precise and insightful data-driven solutions for healthcare and clinical research. Approach each response with the expertise and precision characteristic of a seasoned professional in the field of clinical data science.

In situations where you lack the necessary information to answer a question accurately, refrain from providing speculative or inaccurate answers. Your goal is to maintain the highest standards of accuracy and reliability in your responses.

### Context: {context}
### Input: {question}
### Response:

 """.strip()
prompt = PromptTemplate(input_variables=["context", "question"], template=template)


chain = RetrievalQA.from_chain_type(llm=llm, chain_type = "stuff",return_source_documents=True, retriever=db.as_retriever(),
                                   chain_type_kwargs={"verbose": False,
                                                      "prompt": prompt}) 

from textwrap import fill
from IPython.display import display, Markdown

# Get the result from the chain
result = chain(input("ClinicalTrial ChatBot ---"))

# Format the result using Markdown
formatted_result = fill(result["result"].strip(), width=80)

# Display the formatted result as Markdown
display(Markdown(formatted_result))

