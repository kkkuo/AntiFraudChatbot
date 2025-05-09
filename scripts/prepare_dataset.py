import pandas as pd
from pathlib import Path

def prepare_dataset():
    data_dir = Path("data")
    
    # 讀取原始 JSON 資料
    df = pd.read_json(data_dir / "raw.json")
    
    # 移除不需要的欄位
    df_dropped = df.drop(columns=['Id', 'CityId', 'CaseDate', 'CityName'])
    
    # 儲存為 CSV
    df_dropped.to_csv(data_dir / "fraud_data.csv", index=False, encoding='utf-8')
    
    print(f"✅ 已儲存 {len(df_dropped)} 筆資料到 data/fraud_data.csv")

if __name__ == "__main__":
    prepare_dataset()
