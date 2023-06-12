import matplotlib.pyplot as plt
import pandas as pd

logs = pd.read_csv('train.log')

plt.figure(figsize=(10, 5))

# 第一个子图
plt.subplot(1, 2, 1)  # 添加子图  1*2大小
plt.plot(logs['train_loss'], label='train')
plt.plot(logs['eval_loss'], label='eval')
plt.legend()  # 添加图例
plt.title('loss')
plt.xlabel('epoch')

# 第二个子图
plt.subplot(1, 2, 2)  # 1*2大小
plt.plot(logs['train_acc'], label='train')
plt.plot(logs['eval_acc'], label='eval')
plt.legend()
plt.title('acc')  # 标题
plt.xlabel('epoch')  # x轴

plt.tight_layout()
plt.savefig('curve1-2.png')