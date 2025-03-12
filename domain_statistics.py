# 統計各個domain的題目數量
# 並輸出圖表

import pandas as pd
import os
import matplotlib.pyplot as plt


def main():
    # 讀取parquet檔案
    df = pd.read_csv("data/test-000000-000001.csv")
    # 統計各個domain的題目數量
    domain_counts = df["Domain"].value_counts()
    print(domain_counts)
    # 畫圖
    plt.figure(figsize=(12, 6))
    ax = domain_counts.plot(kind="bar")
    plt.title("Domain Statistics", pad=20)
    plt.xlabel("Domain")
    plt.ylabel("Count")

    # 旋轉x軸標籤並調整對齊方式
    plt.xticks(rotation=45, ha='right')

    # 調整圖表邊距
    plt.tight_layout()

    plt.savefig("assets/domain_statistics.png", bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    main()
