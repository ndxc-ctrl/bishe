import json
import glob
import os

def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))

    # 匹配当前目录下所有 json 文件（排除已经合并好的 uavon_dataset.json）
    pattern = os.path.join(data_dir, "*.json")
    files = sorted(
        f for f in glob.glob(pattern)
        if os.path.basename(f) != "uavon_dataset.json"
    )

    if not files:
        print("当前目录没有找到要合并的 JSON 文件，请先把从 HuggingFace 下载的 JSON 放到这个目录。")
        return

    all_data = []
    for path in files:
        print(f"Loading {path} ...")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            all_data.extend(data)
        else:
            # 如果单个文件本身就是一个 dict，而不是 list，就放到列表里
            all_data.append(data)

    out_path = os.path.join(data_dir, "uavon_dataset.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print("-" * 60)
    print(f"合并完成：共 {len(files)} 个文件，总样本数 {len(all_data)}")
    print(f"已保存到：{out_path}")

if __name__ == "__main__":
    main()