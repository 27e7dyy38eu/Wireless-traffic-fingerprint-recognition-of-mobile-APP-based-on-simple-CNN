import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix



class CSVDataset(Dataset):
    """
    自定义数据集类，用于加载 CSV 文件并将其转换为 PyTorch 张量。
    """
    def __init__(self, root_folder, label_dict, max_length=None):
        """
        :param root_folder: 包含子文件夹的根文件夹路径（如 'data/'）
        :param label_dict: 标签字典，例如 {'normal': 0, 'attack': 1}
        :param max_length: 输入序列的最大长度（用于填充或截断）
        """
        self.data = []
        self.labels = []
        self.max_length = max_length
        for label_name in os.listdir(root_folder):
            label_folder = os.path.join(root_folder, label_name)
            if not os.path.isdir(label_folder):
                continue
            if label_name not in label_dict:
                print(f"警告：未知标签 '{label_name}'，跳过该文件夹。")
                continue
            label = label_dict[label_name]
            for file_name in os.listdir(label_folder):
                if not file_name.endswith('.csv'):
                    continue
                file_path = os.path.join(label_folder, file_name)
                df = pd.read_csv(file_path, encoding='gbk')
                features = df.values.astype(np.float32)
                if self.max_length:
                    if len(features) < self.max_length:
                        padding = np.zeros((self.max_length - len(features), features.shape[1]))
                        features = np.vstack([features, padding])
                    else:
                        features = features[:self.max_length]
                self.data.append(features)
                self.labels.append(label)
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]



class SimpleCNN(nn.Module):
    """
    简单的 CNN 模型，用于二分类任务。
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
        dummy_input = torch.zeros(1, max_length, 2)  
        dummy_input = dummy_input.permute(0, 2, 1)  
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



class EarlyStopping:
    """早停法"""
    def __init__(self, patience=3, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def evaluate_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    accuracy = 100 * np.sum(np.diag(cm)) / np.sum(cm)
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    early_stopping = EarlyStopping(patience=3, verbose=True)
    model.to(device)
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        accuracy = evaluate_model(model, test_loader, device)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "Model/your_model_1.pth")
            print(f"Best model saved with accuracy: {best_accuracy:.2f}%")
        early_stopping(running_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break



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
    
    root_folder = "TAM"
    label_dict = read_label_dict('label.txt')
    print(label_dict)
    max_length = 50  
    batch_size = 4
    num_epochs = 100
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = CSVDataset(root_folder, label_dict, max_length=max_length)
    if len(dataset) == 0:
        raise ValueError("数据集为空，请检查文件夹路径和文件内容！")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_channels = 2
    num_classes = len(label_dict)
    model = SimpleCNN(input_channels, num_classes, max_length)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=num_epochs, device=device)