# 統計各個domain的題目數量
# 並輸出圖表

import pandas as pd
import os
import matplotlib.pyplot as plt


def main():
    # 讀取parquet檔案
    df_normal = pd.read_csv("QA_Data/test-000000-000001.csv")
    df_difficult = pd.read_csv("QA_Data/difficult-000000-000001.csv")
    # 統計各個domain的題目數量
    domain_counts_normal = df_normal["Domain"].value_counts()
    domain_counts_difficult = df_difficult["Domain"].value_counts()
    print(domain_counts_normal)
    print(domain_counts_difficult)
    # 畫圖normal
    plt.figure(figsize=(12, 6))
    ax = domain_counts_normal.plot(kind="bar")
    plt.title("Domain Statistics", pad=20)
    plt.xlabel("Domain")
    plt.ylabel("Count")

    # 旋轉x軸標籤並調整對齊方式
    plt.xticks(rotation=45, ha='right')

    # 調整圖表邊距
    plt.tight_layout()

    plt.savefig("assets/normal_domain_statistics.png",
                bbox_inches='tight', dpi=300)

    # 畫圖difficult
    plt.figure(figsize=(12, 6))
    ax = domain_counts_difficult.plot(kind="bar")
    plt.title("Domain Statistics", pad=20)
    plt.xlabel("Domain")
    plt.ylabel("Count")

    # 旋轉x軸標籤並調整對齊方式
    plt.xticks(rotation=45, ha='right')

    # 調整圖表邊距
    plt.tight_layout()

    plt.savefig("assets/difficult_domain_statistics.png",
                bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    main()
