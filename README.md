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

