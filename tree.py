import os
import sys

def generate_tree(start_path, prefix=""):
    """
    递归生成目录树结构
    """
    items = sorted(os.listdir(start_path))
    pointers = ['├── '] * (len(items) - 1) + ['└── ']

    tree_str = ""

    for pointer, item in zip(pointers, items):
        path = os.path.join(start_path, item)
        tree_str += prefix + pointer + item + "\n"

        if os.path.isdir(path):
            extension = '│   ' if pointer == '├── ' else '    '
            tree_str += generate_tree(path, prefix + extension)

    return tree_str


if __name__ == "__main__":
    target_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")

    if not os.path.exists(target_dir):
        print("目录不存在:", target_dir)
    else:
        print(f"目录结构: {target_dir}\n")

        tree = generate_tree(target_dir)
        print(tree)

        # 可选：保存到文件
        output_file = "structure.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"{target_dir}\n")
            f.write(tree)

        print(f"\n已保存到 {output_file}")
