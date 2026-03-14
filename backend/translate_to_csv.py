

import csv

# 原始資料檔名
input_file = "flag_data.txt"  # 內容可以直接貼你的文字空格電碼
output_file = "mapping.csv"

pairs = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue  # 空行跳過
        items = line.split()
        # 每兩個一組
        for i in range(0, len(items)-1, 2):
            char = items[i]
            code = items[i+1]
            pairs.append((code, char))

# 寫入 CSV
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    for code, char in pairs:
        writer.writerow([code, char])

print("mapping.csv 轉換完成")
