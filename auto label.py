from pathlib import Path
from ultralytics import YOLO
import shutil

def auto_label(model_path, input_dir, output_dir):
    """使用指定模型自動標註圖片"""
    model = YOLO(model_path)  # 載入已訓練的模型
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) / "auto_labels"
    
    # 建立輸出目錄
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    predict_dir = output_dir / "predict"
    predict_dir.mkdir(parents=True, exist_ok=True)
    
    # 取得所有圖片
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    print(f"找到 {len(image_files)} 張圖片")
    
    # 使用模型進行預測
    results = model.predict(
        source=str(input_dir),
        save=True,
        save_txt=True,
        project=str(output_dir),
        name="predict",
        exist_ok=True,
        conf=0.25
    )
    
    # 移動標註文件到正確位置
    predict_labels = output_dir / "predict" / "labels"
    if predict_labels.exists():
        print("移動標註文件到正確位置...")
        moved_count = 0
        for txt in predict_labels.glob("*.txt"):
            if txt.name == "classes.txt":
                continue
            target = labels_dir / txt.name
            txt.rename(target)
            moved_count += 1
        
        shutil.rmtree(predict_labels)  # 清理臨時目錄
        print(f"已移動 {moved_count} 個標註文件到 {labels_dir}")
        print(f"預測結果保存在: {output_dir}/predict")
    
    # 確認標註結果
    total_labels = len(list(labels_dir.glob("*.txt")))
    if total_labels == 0:
        print("警告: 未找到任何標註文件!")
    else:
        print(f"\n自動標註完成!")
        print(f"標註文件數量: {total_labels}")
        print(f"標註檔案位置: {labels_dir}")
        print(f"預測可視化結果: {output_dir}/predict")

def get_latest_model(model_dir):
    """從 model 資料夾中獲取最新的 best.pt"""
    model_dir = Path(model_dir)
    best_models = list(model_dir.glob("*/weights/best.pt"))  # 找出所有 best.pt
    if not best_models:
        raise FileNotFoundError("未找到任何模型 best.pt!")

    # 根據文件的修改時間選擇最新的模型
    latest_model = max(best_models, key=lambda x: x.stat().st_mtime)
    print(f"使用最新模型: {latest_model}")
    return latest_model


if __name__ == '__main__':
    # 模型路徑
    model_path = get_latest_model("./model")
    
    # 輸入資料夾 (待標註的圖片資料夾)
    input_dir = Path("./R20")
    
    # 輸出資料夾 (保存標註結果的資料夾)
    output_dir = Path("./R20")
    
    auto_label(model_path, input_dir, output_dir)