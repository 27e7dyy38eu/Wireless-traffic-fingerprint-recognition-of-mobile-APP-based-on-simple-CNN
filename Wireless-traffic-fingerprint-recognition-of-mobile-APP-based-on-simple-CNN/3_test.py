import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import torch.nn.functional as F  



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



def predict(model, file_path, label_dict, max_length, input_channels, device):
    """
    使用训练好的模型对单个 CSV 文件进行预测，并输出每个类别的概率。
    """
    
    features = preprocess_csv(file_path, max_length, input_channels)
    features = features.to(device)
    with torch.no_grad():
        outputs = model(features)
        probabilities = F.softmax(outputs, dim=1)  
        _, predicted = torch.max(probabilities, 1)
    idx_to_label = {v: k for k, v in label_dict.items()}
    print("\n【预测概率】")
    for idx in range(len(label_dict)):
        class_name = idx_to_label.get(idx, "Unknown")
        prob = probabilities[0][idx].item()
        print(f"{class_name}: {prob:.4f}")
    predicted_label = predicted.item()
    return idx_to_label.get(predicted_label, "Unknown")

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
    
    model_path = "Model/example_model.pth"
    label_dict = read_label_dict('label.txt')
    print(label_dict)
    max_length = 50                        
    input_channels = 2                     
    num_classes = len(label_dict)          
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, input_channels, num_classes, max_length, device)

    csv_file = input("请输入 CSV 文件路径：").strip()
    if not os.path.isfile(csv_file):
        print("文件不存在，请检查路径！")
    else:
        predicted_class = predict(model, csv_file, label_dict, max_length, input_channels, device)
        print(f"\n 最终预测类别：{predicted_class}")