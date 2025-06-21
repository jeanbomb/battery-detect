import os
import random
from shutil import copyfile, rmtree
from ultralytics import YOLO
from pathlib import Path

# 確保 output_dir 定義在全域作用域
output_dir = Path("資料夾位置")  # 替換為實際的資料夾位置
def setup_and_split_dataset():
    """自動清理舊數據並分割新數據集"""
    # 設定數據來源資料夾
    base_dir1 = Path("來源資料夾位置")  # 替換為實際的來源資料夾位置
    base_dir2 = Path("來源資料夾位置2")  # 替換為實際的來源資料夾位置2
    input_dirs = [base_dir1 / f"L{i}" for i in range(1, 14)] + [base_dir2 / f"R{j}" for j in range(14, 21)]


    # 設定輸出資料夾
   
    train_val_test_dirs = [output_dir / "train", output_dir / "val", output_dir / "test"]

    # **清除舊的 train/val/test 資料夾**
    for dir_path in train_val_test_dirs:
        if dir_path.exists():
            print(f"正在刪除舊資料夾: {dir_path}")
            rmtree(dir_path)

    # 訓練/驗證/測試集比例
    train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1

    # 創建新資料夾結構
    for d in train_val_test_dirs:
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "labels").mkdir(parents=True, exist_ok=True)

    # 收集所有圖片與標註文件
    all_images = []
    all_labels = {}
    for input_dir in input_dirs:
        images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
        for img in images:
            label_file = img.with_suffix(".txt")
            if label_file.exists():
                all_images.append(img)
                all_labels[img.name] = label_file

    # 隨機分配數據集
    random.shuffle(all_images)
    num_train = int(train_ratio * len(all_images))
    num_val = int(val_ratio * len(all_images))
    num_test = len(all_images) - num_train - num_val

    datasets = {
        "train": all_images[:num_train],
        "val": all_images[num_train:num_train + num_val],
        "test": all_images[num_train + num_val:]
    }

    # 複製文件到對應資料夾
    for split, images in datasets.items():
        for img in images:
            label_file = all_labels.get(img.name)
            if label_file:
                copyfile(img, output_dir / split / "images" / img.name)
                copyfile(label_file, output_dir / split / "labels" / label_file.name)

    print(f"\n數據集分配完成！")
    print(f"訓練集: {num_train} 張圖片")
    print(f"驗證集: {num_val} 張圖片")
    print(f"測試集: {num_test} 張圖片")

    # 生成 data.yaml 配置文件
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {output_dir}\n")
        f.write(f"train: train\n")
        f.write(f"val: val\n")
        f.write(f"test: test\n")
        f.write(f"nc: 9\n")  # 類別數量
        f.write(f"names: ['1', '2', '3', '4', '5', '6', '9', 'n', 'ng']\n")

    print(f"\ndata.yaml 已生成，路徑: {yaml_path}")
    return yaml_path



def train_model(yaml_path):
    """訓練模型主函數"""
    print("\n開始訓練模型...")
    model = YOLO('yolov8s.pt')  # 使用 YOLOv8s 模型進行訓練
    results = model.train(
        data=str(yaml_path),
        epochs=300,                # 訓練輪數
        imgsz=640,                 # 圖片大小
        batch=8,                   # 批次大小
        conf=0.35,     # 信心閾值設為 0.35，保持對小物件的靈敏度
        iou=0.03,      # 調整 NMS 的 IoU 閾值
        augment=True,  # 增強資料
        name="battery_det_large",  # 訓練名稱
        project=str(output_dir),   # 儲存到 model 資料夾
        device=0,                  # 使用 GPU 0
        workers=4,                 # 增加資料載入進程數
        cache=True                 # 啟用快取
    )
    print(f"\n訓練完成！模型儲存在: {results.save_dir}")

if __name__ == "__main__":
    # 清理舊數據並分割新數據集
    yaml_path = setup_and_split_dataset()
    # 開始訓練模型
    train_model(yaml_path)