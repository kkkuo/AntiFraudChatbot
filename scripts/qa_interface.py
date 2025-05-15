from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# 全域變數來儲存 chain 和 memory
_conversation_chain = None
_memory = None

def load_conversational_retrieval_chain():
    global _conversation_chain
    global _memory
    if _conversation_chain is None:
        # 模型名稱
        model_id = "ziqingyang/chinese-alpaca-2-7b"

        # 載入 tokenizer 和模型
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto"
        )

        # 建立 text-generation pipeline
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, do_sample=True)

        # 包裝成 LangChain 的 LLM
        llm = HuggingFacePipeline(pipeline=pipe)

        # 建立向量嵌入器
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

        # 載入 FAISS 資料庫
        try:
            vectorstore = FAISS.load_local("faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)
            retriever = vectorstore.as_retriever()
        except Exception as e:
            print(f"載入 FAISS 資料庫失敗: {e}")
            return None

        # 建立記憶體物件 (如果還沒有)
        if _memory is None:
            _memory = ConversationBufferMemory(return_messages=True)

        PROMPT_TEMPLATE = """你是一位專業的反詐騙諮詢助手。請根據提供的背景資料和先前的對話歷史，清楚地回答使用者的問題。
        如果找到相關詐騙資訊，請告訴使用者「這很有可能是詐騙」，並提供一則具體的詐騙案例或相關資訊。請務必提供具體的詐騙案例或相關資訊，讓使用者能夠理解為什麼這是詐騙。
        如果使用者還是認為這不是詐騙，請再提供一則具體的詐騙案例或相關資訊，並建議使用者撥打165反詐騙專線。
        如果使用者詢問的問題與詐騙無關，請告訴使用者「這不是詐騙」，並提供具體的理由或建議。請務必提供具體的理由或建議，讓使用者能夠理解為什麼這不是詐騙。
        如果找不到相關資訊，請建議使用者撥打165反詐騙專線，並且告訴使用者「這很有可能是詐騙」。

        背景資料：
        {context}

        使用者問題：
        {question}

        請用繁體中文回答，語氣親切且專業。
        如果有人詢問與詐騙無關的問題，請回覆：「請你去找別人聊天，不要佔用公共資源。」
        """

        PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=PROMPT_TEMPLATE
        )

        # 建立 ConversationalRetrievalChain (如果還沒有)
        _conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=_memory,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=True,
            verbose=False  # 預設不顯示中間步驟，可以根據需要調整
        )
    return _conversation_chain

def run_qa(query, chat_history): # 修改 run_qa 接受 chat_history
    conversation_chain = load_conversational_retrieval_chain()
    if conversation_chain:
        result = conversation_chain.invoke({"question": query, "chat_history": chat_history})
        answer = result["answer"]
        return answer, result["chat_history"]  # ⬅️ 同時回傳 answer 和 chat_history
    else:
        print("警告：ConversationalRetrievalChain 初始化失敗。")
        return "系統初始化失敗", chat_history


