# 🛡️ Anti-Fraud Chatbot

本專案是一個中文防詐騙聊天機器人，使用 FAISS + LangChain + 中文語言模型來實作詐騙案例查詢。用戶可輸入自然語言描述疑似詐騙情境，系統將從資料庫中找出最相似的案例並進行回答。

---

## 📁 專案架構

```bash
.
├── data/
│   ├── raw.json              # 原始詐騙資料（從 165 網站抓取）
│   └── fraud_data.csv        # 處理後的清理資料
│
├── faiss_index/              # 儲存 FAISS 向量索引
│   └── index
│
├── scripts/
│   ├── fetch_data.py         # 從 API 下載詐騙資料
│   ├── prepare_dataset.py    # 處理資料為 CSV
│   ├── embed_and_index.py    # 嵌入文字並儲存 FAISS
│   ├── search_interface.py   # CLI 查詢相似案例
│   └── qa_interface.py       # 啟動 LLM + FAISS 的問答系統
│
├── notebooks/
│   └── main_workflow.ipynb   # Colab 使用流程筆記
└── README.md
🚀 安裝與執行
1️⃣ 安裝套件
bash
Copy
Edit
pip install -r requirements.txt
或在 Colab 中手動安裝：

python
Copy
Edit
!pip install sentence-transformers langchain langchain-community faiss-cpu
2️⃣ 執行流程
bash
Copy
Edit
# 下載資料
python scripts/fetch_data.py

# 處理成 CSV
python scripts/prepare_dataset.py

# 建立向量資料庫
python scripts/embed_and_index.py

# 啟動問答系統查詢
python scripts/qa_interface.py
🤖 問答介面範例
python
Copy
Edit
from scripts.qa_interface import run_qa

run_qa("有人打來說我帳戶異常，請我去 ATM 操作，這是真的嗎？")
🧠 使用模型
嵌入模型：sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

向量資料庫：FAISS

中文語言模型：ziqingyang/chinese-alpaca-2-7b

問答鏈：使用 LangChain 的 RetrievalQA

📊 資料來源
165 全民防騙儀表板

🔧 TODO
 自動定期更新最新案例

 整合 Streamlit 前端

 加入分類模型輔助回應建議

📄 License
本專案僅用於學術與學習用途，請勿用於商業用途或誤導性資訊散播。

