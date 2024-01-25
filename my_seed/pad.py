input_file_path = 'CA-GrQc_LT_seed_output_0.txt'
output_file_path = 'CA-GrQc_IC_seed_output.txt'
target_line_count = 5242

# 读取文件
with open(input_file_path, 'r') as input_file:
    data_lines = input_file.readlines()

# 计算需要补充的行数
lines_to_add = target_line_count - len(data_lines)

# 在原有文件最后一行后添加足够的0
extended_data = data_lines + ['0\n'] * lines_to_add

# 将结果写入新文件
with open(output_file_path, 'w') as output_file:
    output_file.writelines(extended_data)

print(f'处理完成，结果已保存到 {output_file_path}')
