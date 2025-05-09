import requests
import json

def fetch_data(save_path='data/raw.json', verbose=False):
    url = "https://165dashboard.tw/CIB_DWS_API/api/CaseSummary/GetCaseSummaryList"
    headers = {
        "Content-Type": "application/json"
    }

    number_of_per_page = 1000
    page_index = 1
    all_data = []

    payload = {
        "UsingPaging": True,
        "NumberOfPerPage": number_of_per_page,
        "PageIndex": page_index,
        "CaseDate": None,
        "CityId": None,
        "Keyword": None,
        "SearchTermInfos": [],
        "SortOrderInfos": [
            {"SortField": "CaseDate", "SortOrder": "DESC"}
        ]
    }

    while True:
        payload["PageIndex"] = page_index
        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            data = response.json()

            cases = data.get("body", {}).get("Detail", [])
            all_data.extend(cases)

            if verbose:
                print(f"第 {page_index} 頁，取得 {len(cases)} 筆")

            if len(cases) < number_of_per_page:
                break
            page_index += 1
        else:
            print(f"Error: {response.status_code}")
            break

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)

    print(f"✅ 已儲存 {len(all_data)} 筆資料到 {save_path}")

if __name__ == "__main__":
    fetch_data(save_path='data/raw.json', verbose=True)
    print("資料下載完成！")
