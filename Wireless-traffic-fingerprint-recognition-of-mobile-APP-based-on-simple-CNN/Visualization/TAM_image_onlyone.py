import os
import glob
import csv
import matplotlib.pyplot as plt
from matplotlib import rcParams


def read_csv(csv_file):
    """
    读取 CSV 文件并提取上行包和下行包数量。
    :param csv_file: CSV 文件路径
    :return: 上行包数量列表, 下行包数量列表
    """
    up_counts = []
    down_counts = []
    try:
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                if len(row) < 2:
                    raise ValueError(f"CSV 文件格式错误：行 {reader.line_num} 数据不足")
                try:
                    up_count, down_count = map(int, row[:2])  
                    up_counts.append(up_count)
                    down_counts.append(down_count)
                except ValueError:
                    raise ValueError(f"CSV 文件格式错误：行 {reader.line_num} 包含非整数值")
    except FileNotFoundError:
        print(f"文件未找到：{csv_file}")
        return [], []
    except Exception as e:
        print(f"读取 CSV 文件时出错：{e}")
        return [], []
    return up_counts, down_counts



def visualize_folder_data(folder_path, S, labels=None):
    """
    绘制文件夹中所有 CSV 文件的上行包和下行包数量趋势图（下行包数量乘以 -1）。
    :param folder_path: 包含 CSV 文件的文件夹路径
    :param S: 时间间隔（秒）
    :param labels: 每个 CSV 文件对应的标签（用于图例），默认为 None
    """
    
    if not os.path.isdir(folder_path):
        raise ValueError(f"文件夹路径无效：{folder_path}")
    
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))

    if not csv_files:
        raise ValueError(f"文件夹中未找到任何 CSV 文件：{folder_path}")

    if labels is None:
        labels = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]

    elif len(labels) != len(csv_files):
        raise ValueError("标签数量必须与 CSV 文件数量一致")

    rcParams['font.sans-serif'] = ['SimHei']  
    rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 6))

    for i, csv_file in enumerate(csv_files):
        up_counts, down_counts = read_csv(csv_file)
        down_counts = [-count for count in down_counts]
        time_intervals = [j * S for j in range(len(up_counts))]
        plt.plot(time_intervals, up_counts, linestyle="-", marker="o")
        plt.plot(time_intervals, down_counts, linestyle="--", marker="x")
    max_time = max(len(up_counts) * S for up_counts, _ in [read_csv(f) for f in csv_files])
    plt.xlim(0, max_time)
    plt.title("美团流量汇聚矩阵", fontsize=16)
    plt.xlabel("时间 (s)", fontsize=12)
    plt.ylabel("帧数量", fontsize=12)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    plt.tight_layout(rect=[1, 0, 0.85, 1])  
    plt.grid(True)
    plt.savefig("logs/bili.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    folder_path = "../TAM/5美团"
    S = 0.1
    visualize_folder_data(folder_path, S)