import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.font_manager as fm


class SimpleCNN(nn.Module):
    """
    简单的 CNN 模型，用于分类任务。
    """
    def __init__(self, input_channels, num_classes, max_length):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        self._calculate_fc_input(max_length)

        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _calculate_fc_input(self, max_length):
        """动态计算全连接层的输入维度"""
        dummy_input = torch.zeros(1, self.conv1.in_channels, max_length)
        dummy_input = self.pool(torch.relu(self.conv1(dummy_input)))
        dummy_input = self.pool(torch.relu(self.conv2(dummy_input)))
        self.fc_input_dim = dummy_input.view(1, -1).size(1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_model(model_path, input_channels, num_classes, max_length, device):
    model = SimpleCNN(input_channels, num_classes, max_length)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_csv(file_path, max_length, input_channels):
    """
    对单个 CSV 文件进行预处理，返回填充或截断后的张量。
    """
    try:
        df = pd.read_csv(file_path, encoding='gbk')
    except Exception as e:
        raise ValueError(f"无法读取文件 {file_path}: {e}")

    features = df.values.astype(np.float32)

    if features.shape[1] != input_channels:
        raise ValueError(f"CSV 列数应为 {input_channels}，但实际为 {features.shape[1]}")

    if len(features) < max_length:
        padding = np.zeros((max_length - len(features), input_channels))
        features = np.vstack([features, padding])
    else:
        features = features[:max_length]

    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    return features


def predict_and_collect_results(model, file_path, label_dict, max_length, input_channels, device):
    true_label = os.path.basename(os.path.dirname(file_path))
    features = preprocess_csv(file_path, max_length, input_channels)
    features = features.to(device)

    with torch.no_grad():
        outputs = model(features)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(probabilities, 1)

    pred_class_name = label_dict[predicted.item()]
    return true_label, pred_class_name


def read_label_dict(file_path):
    label_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, value = line.split(':', 1)
            label_dict[key.strip()] = int(value.strip())
    return label_dict


if __name__ == "__main__":
    data_dir = "../TAM"
    model_path = "../Model/your_model.pth"
    label_dict = read_label_dict('../label.txt')
    reverse_label_dict = {v: k for k, v in label_dict.items()}
    max_length = 50
    input_channels = 2
    num_classes = len(label_dict)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(model_path, input_channels, num_classes, max_length, device)

    y_true = []
    y_pred = []

    for label in os.listdir(data_dir):
        if label not in label_dict:
            continue
        class_dir = os.path.join(data_dir, label)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(class_dir, file_name)
                    if os.path.isfile(file_path):  # 确保是文件
                        true_label, pred_label = predict_and_collect_results(model, file_path, reverse_label_dict,
                                                                             max_length, input_channels, device)
                        y_true.append(true_label)
                        y_pred.append(pred_label)

    # 设置中文字体
    try:
        # Linux 示例字体路径（根据你的系统修改）
        font_path = '/usr/share/fonts/truetype/arphic/ukai.ttc'
        # Windows 示例字体路径
        # font_path = 'C:\\Windows\\Fonts\\simhei.ttf'
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
    except Exception as e:
        print("加载自定义字体失败，尝试使用默认字体...")
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 显示中文

    plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

    # 生成混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=list(label_dict.keys()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_dict.keys()))
    disp.plot(cmap=plt.cm.GnBu)

    plt.title('混淆矩阵', fontsize=14)
    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.xticks(fontsize=10, rotation=45)
    plt.yticks(fontsize=10, rotation=45)
    plt.tight_layout()

    # 保存图像
    plt.savefig("logs/confusion_matrix.png", dpi=600, bbox_inches='tight')
    plt.show()