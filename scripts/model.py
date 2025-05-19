#HuggingFace models
from langchain_huggingface_hub import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
#Langchain models
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

llm = None
retriever = None
prompt = None

def load_model():
    if llm = None:
        model_id = 'Qwen/Qwen3-0.6B'

