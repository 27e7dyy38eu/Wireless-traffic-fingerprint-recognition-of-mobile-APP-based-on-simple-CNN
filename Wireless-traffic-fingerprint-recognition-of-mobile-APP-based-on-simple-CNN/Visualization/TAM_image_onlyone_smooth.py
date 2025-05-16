import os
import glob
import csv
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.interpolate import make_interp_spline
import numpy as np


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


def visualize_folder_data(folder_path, S):
    """
    绘制文件夹中所有 CSV 文件的上行包和下行包数量趋势图（下行包数量乘以 -1），并使用平滑曲线。
    :param folder_path: 包含 CSV 文件的文件夹路径
    :param S: 时间间隔（秒）
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"文件夹路径无效：{folder_path}")

    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not csv_files:
        raise ValueError(f"文件夹中未找到任何 CSV 文件：{folder_path}")

    rcParams['font.sans-serif'] = ['SimHei']
    rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(12, 6))

    max_time = 0
    for i, csv_file in enumerate(csv_files):
        up_counts, down_counts = read_csv(csv_file)
        down_counts = [-count for count in down_counts]
        time_intervals = [j * S for j in range(len(up_counts))]
        max_time = max(max_time, time_intervals[-1])

        x = np.array(time_intervals)
        y_up = np.array(up_counts)
        y_down = np.array(down_counts)

        if len(x) < 2:
            print(f"文件 {csv_file} 数据不足，跳过插值")
            continue

        try:
            spl_up = make_interp_spline(x, y_up, k=3) 
            xs_up = np.linspace(x.min(), x.max(), 500)
            ys_up = spl_up(xs_up)

            spl_down = make_interp_spline(x, y_down, k=3)
            xs_down = np.linspace(x.min(), x.max(), 500)
            ys_down = spl_down(xs_down)

            plt.plot(xs_up, ys_up, linestyle="-", color='r', label="Uplink" if i == 0 else "", alpha=0.6)
            plt.plot(xs_down, ys_down, linestyle="--", color='r', label="Downlink" if i == 0 else "", alpha=0.6)

        except Exception as e:
            print(f"插值失败，跳过文件 {csv_file}: {e}")
            plt.plot(x, y_up, linestyle="-", color='b', label="Uplink" if i == 0 else "", alpha=0.6)
            plt.plot(x, y_down, linestyle="--", color='b', label="Downlink" if i == 0 else "", alpha=0.6)

    plt.xlim(0, max_time)
    plt.title("美团流量汇聚矩阵", fontsize=16)
    plt.xlabel("时间 (s)", fontsize=12)
    plt.ylabel("帧数量", fontsize=12)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')

    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys(), loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.grid(True)
    plt.savefig("logs/Tiktok.png", dpi=600)
    plt.show()

if __name__ == "__main__":
    folder_path = "../TAM/5美团"
    S = 0.1
    visualize_folder_data(folder_path, S)
