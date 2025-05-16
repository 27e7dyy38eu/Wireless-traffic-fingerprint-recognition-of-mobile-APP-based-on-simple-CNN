import os
import glob
import csv
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.interpolate import make_interp_spline
import numpy as np
from matplotlib.cm import get_cmap

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
            next(reader)  # 跳过表头

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


def visualize_data(data_folder, S, label_colors=None, alpha=0.7):
    """
    绘制 data 文件夹中每个 label 文件夹下的所有 CSV 文件的上行包和下行包数量趋势图（下行包数量乘以 -1）。
    :param data_folder: 包含 label 文件夹的根文件夹路径
    :param S: 时间间隔（秒）
    :param label_colors: 每个 label 对应的颜色字典，默认为 None
    :param alpha: 线条和标记的透明度，取值范围 0 到 1，默认为 0.7
    """
    if not os.path.isdir(data_folder):
        raise ValueError(f"数据文件夹路径无效：{data_folder}")

    labels = [name for name in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, name))]
    if not labels:
        raise ValueError(f"数据文件夹中未找到任何 label 文件夹：{data_folder}")

    if label_colors is None:
        label_colors = {label: 'C{}'.format(i % 10) for i, label in enumerate(labels)}

    rcParams['font.sans-serif'] = ['SimHei']
    rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(14, 7))

    all_time_intervals = []

    for label in labels:
        label_path = os.path.join(data_folder, label)
        csv_files = sorted(glob.glob(os.path.join(label_path, "*.csv")))

        if not csv_files:
            print(f"Label 文件夹中未找到任何 CSV 文件：{label_path}")
            continue

        color = label_colors.get(label, 'black')
        first_file = True  # 标记是否是当前 label 的第一个文件

        for csv_file in csv_files:
            up_counts, down_counts = read_csv(csv_file)

            time_intervals = [j * S for j in range(len(up_counts))]
            all_time_intervals.append(time_intervals)

            down_counts_neg = [-count for count in down_counts]

            # 控制图例仅在第一个文件绘制时添加
            label_up = f"{label}-uplink" if first_file else "_nolegend_"
            label_down = f"{label}-downlink" if first_file else "_nolegend_"

            x = np.array(time_intervals)
            y_up = np.array(up_counts)
            y_down_neg = np.array(down_counts_neg)

            if len(x) >= 2:
                spl_up = make_interp_spline(x, y_up, k=3)
                xs = np.linspace(x.min(), x.max(), 500)
                ys_up = spl_up(xs)

                spl_down = make_interp_spline(x, y_down_neg, k=3)
                ys_down = spl_down(xs)

                plt.plot(xs, ys_up, linestyle="-", marker="o", color=color,
                         label=label_up, linewidth=1.5, markersize=1, alpha=alpha)
                plt.plot(xs, ys_down, linestyle="--", marker="x", color=color,
                         label=label_down, linewidth=1.5, markersize=1, alpha=alpha)
            else:
                plt.plot(x, y_up, linestyle="-", marker="o", color=color,
                         label=label_up, linewidth=1.5, markersize=4, alpha=alpha)
                plt.plot(x, y_down_neg, linestyle="--", marker="x", color=color,
                         label=label_down, linewidth=1.5, markersize=4, alpha=alpha)

            first_file = False  # 后续文件不加图例

    max_time = max(max(t) for t in all_time_intervals) if all_time_intervals else 0

    plt.title("流量汇聚矩阵", fontsize=16)
    plt.xlabel("时间 (s)", fontsize=12)
    plt.ylabel("帧数量", fontsize=12)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=alpha)
    plt.xlim(0, max_time)
    plt.grid(True, linestyle=':', alpha=0.7)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=color, lw=2, linestyle='-', label=f'{label} - Uplink') for label, color in label_colors.items()
    ] + [
        Line2D([0], [0], color=color, lw=2, linestyle='--', label=f'{label} - Downlink') for label, color in label_colors.items()
    ]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plt.savefig("logs/TAM_windows_total_smooth.png", dpi=600, bbox_inches='tight')
    plt.show()


def read_label_dict(file_path):
    label_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # 跳过空行或注释
            key, value = line.split(':', 1)
            label_dict[key.strip()] = int(value.strip())
    return label_dict


if __name__ == "__main__":
    data_folder = "../TAM"  # 替换为你的数据文件夹路径
    label_dict = read_label_dict('../label.txt')
    S = 0.1  # 时间间隔（秒）

    # 获取所有 label 文件夹名称
    labels = [name for name in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, name))]

    # 使用 colormap 为每个 label 分配颜色
    cmap = get_cmap('tab10')  # 可选 'Set3', 'Dark2', 'Pastel1' 等
    label_colors = {label: cmap(i % 10) for i, label in enumerate(labels)}  # %10 防止超出范围

    alpha = 0.5  # 设置透明度

    visualize_data(data_folder, S, label_colors, alpha)