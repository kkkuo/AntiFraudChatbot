# AntiFraudChatbot
A chatbot designed for fraud prevention.
165-fraud-detection/
│
├── data/                    # 存放原始資料與處理後資料
│   ├── raw/                 # 原始 API 抓取下來的 JSON
│   └── processed/           # 處理後轉成 DataFrame 的 CSV 或 JSON
│
├── notebooks/               # Jupyter 或 Colab 筆記本（資料探索與實驗）
│   └── main_workflow.ipynb  # 主流程紀錄與分析
│
├── scripts/                 # 可重複使用的程式碼
│   ├── fetch_data.py        # 抓資料用的程式
│   ├── embed_and_index.py   # 做 embedding 並儲存 FAISS index
│   └── search_interface.py  # 提供查詢（例如 command-line 或簡單介面）
│
├── faiss_index/             # 儲存 FAISS 向量索引檔
│   └── index_file.index     # 實際 index 檔
│
├── README.md                # 專案說明文件
└── .gitignore               # 忽略像是 __pycache__、.ipynb_checkpoints 等
