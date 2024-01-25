
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# 数据
default_rate = [1, 5, 10, 20, 40]

jazz_OPIM_LT_y = [35.5985, 128.793, 161.727, 182.617, 193.781]
jazz_OPIM_IC_y = [21.8932, 71.6333, 95.6975, 124.135, 159.447]

jazz_Deep_LT_y_max = [2,  12,  25,  50, 197]
jazz_Deep_LT_y_min = [1,   9,  19,  39, 119]
jazz_Deep_LT_y = [1.1,  10,   21.1,  42.5, 189.2]

jazz_Deep_IC_y_max = [39.1,  81.8, 102.1, 126.3, 148.9]
jazz_Deep_IC_y_min = [2.1,  53.8,  83.7, 115.3, 141.5]
jazz_Deep_IC_y = [18.04,  71.19,  95.6,  120.34, 146.07]
# 绘图
plt.figure(figsize=(10, 6))

# jazz_OPIM_LT
plt.subplot(1, 2, 1)
plt.plot(default_rate, jazz_OPIM_LT_y, marker='o', label='jazz_OPIM_LT')
plt.plot(default_rate, jazz_Deep_LT_y, marker='o', label='jazz_Deep_LT')
plt.fill_between(default_rate, jazz_Deep_LT_y_min, jazz_Deep_LT_y_max, color='gray', alpha=0.2, label='Deep_LT Range')
plt.title('jazz_OPIM_LT vs jazz_Deep_LT')
plt.xlabel('Default Rate')
plt.ylabel('Values')
plt.legend()

# jazz_OPIM_IC
plt.subplot(1, 2, 2)
plt.plot(default_rate, jazz_OPIM_IC_y, marker='o', label='jazz_OPIM_IC')
plt.plot(default_rate, jazz_Deep_IC_y, marker='o', label='jazz_Deep_IC')
plt.fill_between(default_rate, jazz_Deep_IC_y_min, jazz_Deep_IC_y_max, color='gray', alpha=0.2, label='Deep_IC Range')
plt.title('jazz_OPIM_IC vs jazz_Deep_IC')
plt.xlabel('Default Rate')
plt.ylabel('Values')
plt.legend()

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()

plt.savefig('comparison_plot.png')  # 将图形保存为文件
