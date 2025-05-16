import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties
import matplotlib.cm as cm


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号


class SimpleCNN(nn.Module):
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
        dummy_input = torch.zeros(1, self.conv1.in_channels, max_length)
        dummy_input = self.pool(torch.relu(self.conv1(dummy_input)))
        dummy_input = self.pool(torch.relu(self.conv2(dummy_input)))
        self.fc_input_dim = dummy_input.view(1, -1).size(1)

    def forward(self, x, return_features=False):
        x = x.permute(0, 2, 1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        features = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(features))
        if return_features:
            return x, features
        return x


def load_model(model_path, input_channels, num_classes, max_length, device):
    model = SimpleCNN(input_channels, num_classes, max_length)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_csv(file_path, max_length, input_channels):
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


def predict_and_collect_features(model, file_path, label_dict, max_length, input_channels, device):
    true_label = os.path.basename(os.path.dirname(file_path))
    features = preprocess_csv(file_path, max_length, input_channels)
    features = features.to(device)

    with torch.no_grad():
        _, feature = model(features, return_features=True)
        feature_vector = feature.cpu().numpy()

    return true_label, feature_vector


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
    data_dir = "../TAM"
    model_path = "../Model/your_model_1.pth"
    label_dict = read_label_dict('../label.txt')
    reverse_label_dict = {v: k for k, v in label_dict.items()}
    max_length = 50
    input_channels = 2
    num_classes = len(label_dict)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(model_path, input_channels, num_classes, max_length, device)

    all_features = []
    y_true = []

    for label in os.listdir(data_dir):
        if label not in label_dict:
            continue
        class_dir = os.path.join(data_dir, label)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(class_dir, file_name)
                    true_label, feature = predict_and_collect_features(model, file_path, reverse_label_dict,
                                                                       max_length, input_channels, device)
                    all_features.append(feature)
                    y_true.append(true_label)

    all_features = np.vstack(all_features)

    # 标准化 + PCA
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features)
    pca = PCA(n_components=3)
    features_3d = pca.fit_transform(all_features_scaled)

    # 创建保存目录
    save_dir = "logs"
    os.makedirs(save_dir, exist_ok=True)

    # 绘图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    cmap = cm.get_cmap('tab20')  # 支持更多类别的颜色
    colors = [cmap(i) for i in range(len(label_dict))]

    for idx, (label_name, color) in enumerate(zip(label_dict.keys(), colors)):
        indices = [i for i, l in enumerate(y_true) if l == label_name]
        if len(indices) > 0:
            ax.scatter(
                features_3d[indices, 0],
                features_3d[indices, 1],
                features_3d[indices, 2],
                c=[color],
                label=label_name,
                s=20
            )

    ax.set_title("特征空间三维分布（PCA）")
    ax.set_xlabel("主成分1")
    ax.set_ylabel("主成分2")
    ax.set_zlabel("主成分3")
    ax.legend(title="类别标签", bbox_to_anchor=(1.05, 1), loc='upper left')

    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "logs/feature_distribution_3d.png"), dpi=600, bbox_inches='tight')
    plt.show()