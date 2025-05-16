import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os
import matplotlib.font_manager as fm


def setup_chinese_font():
    font_path = "C:/Windows/Fonts/simhei.ttf"
    if os.path.exists(font_path):
        font = fm.FontProperties(fname=font_path, size=12)
        plt.rcParams['font.sans-serif'] = [font.get_name()]
    else:
        print("未找到自定义字体，使用备用字体...")
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']

    plt.rcParams['axes.unicode_minus'] = False
setup_chinese_font()
log_file = "logs/training_log.txt"
if not os.path.exists(log_file):
    raise FileNotFoundError(f"日志文件 {log_file} 不存在，请检查路径是否正确。")
epochs = []
train_losses = []
train_accuracies = []
test_accuracies = []

with open(log_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines[2:]:
    if not line.strip():
        continue
    parts = line.split("|")
    if len(parts) < 4:
        continue
    try:
        epoch = int(parts[0].strip())
        loss = float(parts[1].strip())
        train_acc = float(parts[2].strip().replace('%', ''))
        test_acc = float(parts[3].strip().replace('%', ''))
        epochs.append(epoch)
        train_losses.append(loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
    except Exception as e:
        print(f"解析行时出错: {line}, 错误信息: {e}")

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.set_xlabel('训练轮次 (Epoch)')
ax1.set_ylabel('损失值 (Loss)', color='tab:blue')
ax1.plot(epochs, train_losses, label='训练损失', color='tab:blue', linewidth=2, alpha=0.7)
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True, linestyle='--', alpha=0.5)

ax2 = ax1.twinx()
ax2.set_ylabel('准确率 (%)', color='tab:green')
ax2.plot(epochs, train_accuracies, label='训练准确率', color='tab:green', linestyle='-', linewidth=2, alpha=0.7)
ax2.plot(epochs, test_accuracies, label='测试准确率', color='tab:orange', linestyle='--', linewidth=2, alpha=0.7)
ax2.tick_params(axis='y', labelcolor='tab:green')

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right')


plt.title('训练过程中的损失与准确率变化')
plt.tight_layout()
plt.savefig("logs/training_performance_combined.png", dpi=600, bbox_inches='tight')
plt.show()