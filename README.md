# 🛡️ Anti-Fraud Chatbot

本專案是一個中文防詐騙聊天機器人，使用 FAISS + LangChain + 中文語言模型來實作詐騙案例查詢。用戶可輸入自然語言描述疑似詐騙情境，系統將從資料庫中找出最相似的案例並進行回答。

---

## 📁 專案架構

```bash
├── data/
│   ├── raw.json              # 原始詐騙資料（從 165 網站抓取）
│   └── fraud_data.csv        # 處理後的清理資料
│
├── faiss_index/              # 儲存 FAISS 向量索引
│   └── index.faiss
│   └── index.pkl
│
├── scripts/
│   ├── fetch_data.py             # 從 API 下載詐騙資料
│   ├── prepare_dataset.py        # 處理資料為 CSV
│   ├── embed_and_index.py        # 嵌入文字並儲存 FAISS
│   ├── model_gemini_version.py   # 啟動 LLM + FAISS 的問答
│   └── app_g.py                  # 啟動網頁互動式介面
│
├── requirements.txt
└── README.md

```

## 🚀 安裝與執行

### 1️⃣ 安裝套件

```bash
pip install -r requirements.txt
```

### 2️⃣ 執行流程

#### 抓取最新詐騙案例資料

如果不需抓取最新的資料則不需執行。請注意，如果多次執行本段 script 會有 IP 被封鎖的風險。

```
python scripts/fetch_data.py
```

#### 對抓下來的資料進行前處理

將資料留下需要的欄位，並存成 csv 檔。

```
python scripts/prepare_dataset.py
```

#### 建立向量資料庫

透過 LangChain 套件，將前一步驟擷取的資料建立為 FAISS vector database。

```
python scripts/embed_and_index.py
```

#### 建立多輪對話聊天系統

結合先前建置的向量資料庫與生成式模型，完成 RAG 應用。

```
python scripts/model.py
```

## 🤖 啟動聊天界面

使用 gradio demo 互動型 chatbot。

```
!python scripts/app_g.py
```

## 🧠 使用模型

嵌入模型：BAAI/bge-m3

向量資料庫：FAISS

中文語言模型：Gemini-2.0-flash

展示介面：Gradio

## 📊 資料來源

165 全民防騙儀表板
