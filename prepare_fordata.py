import os
import shutil
import random

# Kaggle解压后的图片目录
source_dir = "/home/hyb/.cache/kagglehub/competitions/dogs-vs-cats/train"

# 新数据集目录
target_dir = "pytorch_pratices/train_data/cats_dogs"

classes = ["cat", "dog"]

# 创建目录
for split in ["train", "val", "test"]:
    for class_name in classes:
        os.makedirs(
            os.path.join(target_dir, split, class_name),
            exist_ok=True
        )

# 分别处理猫和狗
for class_name in classes:

    files = [
        f for f in os.listdir(source_dir)
        if f.startswith(class_name + ".")
    ]

    # 打乱顺序
    random.shuffle(files)

    # 这里先只取3000张，方便学习
    files = files[:3000]

    # 80% / 10% / 10%
    train_end = int(len(files) * 0.8)
    val_end = int(len(files) * 0.9)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    for split, split_files in splits.items():

        for filename in split_files:

            src = os.path.join(source_dir, filename)

            dst = os.path.join(
                target_dir,
                split,
                class_name,
                filename
            )

            shutil.copy(src, dst)

print("数据集整理完成！")