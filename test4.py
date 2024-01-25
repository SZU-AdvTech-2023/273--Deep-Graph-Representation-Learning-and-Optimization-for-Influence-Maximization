import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# 定义用于存储提取数据的列表
epochs = []
totals = []

# 读取文本文件并提取数据
with open('your_file.txt', 'r') as file:
    for line in file:
        if line.startswith('Epoch'):
            parts = line.split()
            epoch_value = parts[1]
            total_value = parts[3]
            if epoch_value.isdigit() and total_value.replace('.', '').isdigit():
                epochs.append(int(epoch_value))
                totals.append(float(total_value))

# 绘图
plt.plot(epochs, totals, label='Total')
plt.title('Total vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Total')
plt.legend()
plt.savefig('output.png')
