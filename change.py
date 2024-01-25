data = '''
13
12
44
0
50
37
28
33
29
24
39
8

'''

# 将字符串按行分割成列表
numbers = data.strip().split('\n')

# 输入百分比，选择前百分之几的数据
percentage = 50
selected_count = int(len(numbers) * percentage / 100)
selected_numbers = numbers[:selected_count]

# 在每两个数字之间加上逗号
result = ','.join(selected_numbers)

# 保存为新的文件
output_filename = f"seed_{percentage}.txt"
with open(output_filename, 'w') as file:
    file.write(result)

print(f"Selected data has been written to {output_filename}")
