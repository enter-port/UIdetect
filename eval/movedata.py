import os
import shutil
import json

def copy_test_files(json_file_path, base_dir, target_xml_folder, target_png_folder):
    # 创建目标文件夹（如果它们不存在）
    os.makedirs(target_xml_folder, exist_ok=True)
    os.makedirs(target_png_folder, exist_ok=True)

    # 加载.json文件内容
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # 遍历测试集路径列表
    for path in data.get('test', []):
        full_path = os.path.join(base_dir, path.lstrip('/'))  # 去除开头的斜杠，以防路径问题
        
        # 构建源文件路径
        source_xml = f"{full_path}.xml"
        source_png = f"{full_path}.png"

        # 获取原始的文件名（不修改文件名）
        original_base_name = os.path.basename(full_path)  # 获取文件名（不含路径）
        
        # 构建目标文件路径
        target_xml = os.path.join(target_xml_folder, f"{original_base_name}.xml")
        target_png = os.path.join(target_png_folder, f"{original_base_name}.png")

        # 复制文件到目标文件夹，并处理可能的错误
        try:
            if os.path.exists(source_xml):
                shutil.copy2(source_xml, target_xml)  # 使用copy2以保留元数据
                print(f"Copied {original_base_name}.xml successfully.")
            else:
                print(f"Source XML file not found: {source_xml}")

            if os.path.exists(source_png):
                shutil.copy2(source_png, target_png)
                print(f"Copied {original_base_name}.png successfully.")
            else:
                print(f"Source PNG file not found: {source_png}")
        except Exception as e:
            print(f"An error occurred while copying {original_base_name} files: {e}")

    print("Process completed.")

# 指定你的.json文件路径、基础路径以及目标文件夹路径
json_file_path = './data/train_test_split.json'  # 替换为你的.json文件路径
base_dir = './data'  # 替换为包含测试数据的基础目录路径
target_xml_folder = '../eval/test_xml'  # 替换为.xml文件的目标文件夹路径
target_png_folder = '../eval/test_pics'  # 替换为.png文件的目标文件夹路径

# 执行函数
copy_test_files(json_file_path, base_dir, target_xml_folder, target_png_folder)
