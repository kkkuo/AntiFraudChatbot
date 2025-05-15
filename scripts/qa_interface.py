from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.combine_documents import StuffDocumentsChain  # ✅ 加這個
from langchain.memory import ConversationBufferMemory

# 全域變數：只初始化一次
_conversation_chain = None

def load_conversational_retrieval_chain():
    global _conversation_chain
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
        llm = HuggingFacePipeline(pipeline=pipe)

        # 向量嵌入器
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

        # 載入 FAISS 資料庫
        try:
            vectorstore = FAISS.load_local("faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)
            retriever = vectorstore.as_retriever()
        except Exception as e:
            print(f"❌ 載入 FAISS 資料庫失敗: {e}")
            return None

        # ✅ Prompt 模板（不用 chat_history）
        PROMPT_TEMPLATE = """你是一位專業的反詐騙諮詢助手。請根據提供的背景資料和對話歷史，清楚地回答使用者的問題。
如果找到相關詐騙資訊，請告訴使用者「這很有可能是詐騙」，並提供一則具體的詐騙案例或相關資訊。
如果使用者還是認為這不是詐騙，請再提供一則案例，並建議撥打165反詐騙專線。
如果問題與詐騙無關，請說「這不是詐騙」，並提供具體理由。
如果找不到資訊，請建議撥打165專線，並說「這很有可能是詐騙」。

背景資料：
{context}

使用者問題：
{question}

請用繁體中文回答，語氣親切且專業。
如果有人詢問與詐騙無關的問題，請回覆：「請你去找別人聊天，不要佔用公共資源。」
"""

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=PROMPT_TEMPLATE
        )

        # ✅ 包裝成 StuffDocumentsChain（這才是 ConversationalRetrievalChain 要的格式）
        combine_docs_chain = StuffDocumentsChain(llm=llm, prompt=prompt)

        # ✅ 加入記憶體
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=False
        )

        # ✅ 建立完整 Chain
        _conversation_chain = ConversationalRetrievalChain(
            retriever=retriever,
            combine_docs_chain=combine_docs_chain,
            memory=memory,
            return_source_documents=True,
            verbose=False
        )

    return _conversation_chain
