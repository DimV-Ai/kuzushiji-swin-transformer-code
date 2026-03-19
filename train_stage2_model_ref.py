import os
import argparse
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)
from torchvision.transforms import (
    Compose, Normalize, RandomResizedCrop, Resize, ToTensor, 
    RandomRotation, ColorJitter, RandomAffine
)
import evaluate

# ================= 設定エリア =================
DATASET_DIR = "/home/yoshiga/research_projects/kuzushiji_clean/kuzushiji_classifier_data_ogi"
# MODEL_CHECKPOINT = "microsoft/swin-tiny-patch4-window7-224"
# 修正後（強くてニューゲーム）
MODEL_CHECKPOINT = "microsoft/swin-base-patch4-window7-224"
OUTPUT_DIR = "/home/yoshiga/research_projects/kuzushiji_clean/kuzushiji_classifier_model"

BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 5e-5
# =======================================

# --- グローバル領域 ---
image_processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)
accuracy_metric = evaluate.load("accuracy")

def train_transforms(examples):
    transforms = Compose([
        # 【重要変更】RandomResizedCrop（切り抜き）は廃止！
        # 代わりに Resize で「画像全体」を強制的に224x224にする。
        # 多少縦横比が変わるが、切り抜かれて見えなくなるより100倍マシ。
        Resize(size), 
        
        # 【新規】回転だけでなく、上下左右への「平行移動」と「縮小」も少し許可
        # これで「文字の位置ズレ」に強くなる
        RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        
        # 色変化（これは継続）
        ColorJitter(brightness=0.2, contrast=0.2),
        
        ToTensor(), 
        normalize
    ])
    examples["pixel_values"] = [transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

def val_transforms(examples):
    transforms = Compose([Resize(size), ToTensor(), normalize])
    examples["pixel_values"] = [transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

def parse_args():
    parser = argparse.ArgumentParser(description="Kuzushiji Classifier Training")
    parser.add_argument("--dry-run", action="store_true", help="Enable dry run mode (fast check)")
    return parser.parse_args()

def main():
    args_cli = parse_args()

    if not os.path.exists(DATASET_DIR):
        print(f"エラー: データディレクトリ {DATASET_DIR} が見つかりません。")
        return

    print(f"Loading dataset from: {DATASET_DIR} ...")
    dataset = load_dataset("imagefolder", data_dir=DATASET_DIR)
    
    # ラベルカラム名の統一
    if "label" in dataset["train"].column_names:
        dataset = dataset.rename_column("label", "labels")

    # 【修正箇所】 検証用データがない場合、自動的に分割を作成する
    if "validation" not in dataset and "test" not in dataset:
        print("⚠️ No validation/test split found. Splitting train dataset (90% Train, 10% Test)...")
        # seedを固定して再現性を確保
        dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
        # train_test_splitの結果は {'train': ..., 'test': ...} になる

    # ドライランモードの適用
    if args_cli.dry_run:
        print("\n⚠️  [DRY RUN MODE] ENABLED ⚠️")
        print("   - データセットを縮小します")
        
        # データを縮小
        dataset["train"] = dataset["train"].select(range(min(200, len(dataset["train"]))))
        
        # 検証用データの名前解決 ('test' か 'validation' か)
        val_split_name = "test" if "test" in dataset else "validation"
        dataset[val_split_name] = dataset[val_split_name].select(range(min(50, len(dataset[val_split_name]))))
            
        training_args_config = {
            "max_steps": 10,
            "eval_strategy": "steps",
            "eval_steps": 10,
            "save_strategy": "steps",
            "save_steps": 10,
            "logging_steps": 1,
            "num_train_epochs": 1,
        }
    else:
        print("\n🚀 [PRODUCTION MODE] ENABLED 🚀")
        training_args_config = {
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "logging_steps": 50,
            "num_train_epochs": EPOCHS,
        }

    # クラス情報の取得
    labels_list = dataset["train"].features["labels"].names
    id2label = {str(i): label for i, label in enumerate(labels_list)}
    label2id = {label: str(i) for i, label in enumerate(labels_list)}
    print(f"✅ Found {len(labels_list)} classes.")

    # 検証用スプリット名の確定
    val_split_name = "test" if "test" in dataset else "validation"

    # 前処理の適用
    train_ds = dataset["train"].with_transform(train_transforms)
    val_ds = dataset[val_split_name].with_transform(val_transforms)

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(labels_list),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        remove_unused_columns=False,
        save_total_limit=2,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        dataloader_num_workers=4,
        fp16=True,
        **training_args_config
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DefaultDataCollator(),
        compute_metrics=compute_metrics,
    )

    print("Starting Training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model()
    image_processor.save_pretrained(OUTPUT_DIR)
    print(f"\n✅ Training Completed. Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()