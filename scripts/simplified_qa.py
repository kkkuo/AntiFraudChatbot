from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
# Fix import path for StuffDocumentsChain based on your LangChain version
try:
    # Try newer import path first
    from langchain_core.documents import StuffDocumentsChain
except ImportError:
    try:
        # Try community import path
        from langchain_community.chains.combine_documents import StuffDocumentsChain
    except ImportError:
        # Fall back to original import path for older versions
        try:
            from langchain.chains.combine_documents import StuffDocumentsChain
        except ImportError:
            print("❌ Could not import StuffDocumentsChain from any known location")

_llm = None
_retriever = None
_prompt = None

def load_model_and_components():
    global _llm, _retriever, _prompt
    
    if _llm is None:
        # Load model components
        model_id = "ziqingyang/chinese-alpaca-2-7b"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto"
        )

        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, do_sample=True)
        _llm = HuggingFacePipeline(pipeline=pipe)

    if _retriever is None:
        # Load vector database and retriever
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        try:
            vectorstore = FAISS.load_local("faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)
            _retriever = vectorstore.as_retriever()
        except Exception as e:
            print(f"❌ 載入 FAISS 資料庫失敗: {e}")
            return None, None, None

    if _prompt is None:
        # Define prompt template
        PROMPT_TEMPLATE = """你是一位專業的反詐騙諮詢助手。請根據提供的背景資料和對話歷史，清楚地回答使用者的問題。
                        如果找到相關詐騙資訊，請告訴使用者「這很有可能是詐騙」，並提供一則具體的詐騙案例或相關資訊。
                        如果使用者還是認為這不是詐騙，請再提供一則案例，並建議撥打165反詐騙專線。
                        如果問題與詐騙無關，請說「這不是詐騙」，並提供具體理由。
                        如果找不到資訊，回覆「這樣的情況應該不是詐騙，如果仍有疑惑，建議撥打165專線詢問專員」。

                        對話歷史：
                        {chat_history}

                        背景資料：
                        {context}

                        使用者問題：
                        {question}

                        請用繁體中文回答，語氣親切且專業。
                        如果有人詢問與詐騙無關的問題，請回覆：「請你去找別人聊天，不要佔用公共資源。」
                        """
        
        _prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template=PROMPT_TEMPLATE
        )
    
    return _llm, _retriever, _prompt

def query_system(question, chat_history_str=""):
    """
    A simplified query function that handles retrieval and response generation
    
    Args:
        question (str): The user's question
        chat_history_str (str): String representation of chat history
        
    Returns:
        dict: Response containing answer and source documents
    """
    llm, retriever, prompt = load_model_and_components()
    
    if not all([llm, retriever, prompt]):
        return {"answer": "系統載入失敗，請稍後再試。", "source_documents": []}
    
    # Retrieve relevant documents
    docs = retriever.get_relevant_documents(question)
    
    if not docs:
        return {"answer": "這樣的情況應該不是詐騙，如果仍有疑惑，建議撥打165專線詢問專員」。", "source_documents": []}
    
    # Format context from documents
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create and use LLM chain for single query without StuffDocumentsChain
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    response = llm_chain.invoke({
        "context": context,
        "question": question,
        "chat_history": chat_history_str
    })
    
    # Handle different response formats
    if isinstance(response, dict) and "text" in response:
        answer = response["text"]
    elif isinstance(response, str):
        answer = response
    else:
        answer = str(response)
    
    return {"answer": answer, "source_documents": docs}