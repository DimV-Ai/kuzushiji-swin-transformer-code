import os
import argparse
import torch
import numpy as np
#from datasets import load_dataset
from datasets import load_dataset, Image, Features, Value
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

import csv
from datetime import datetime
from pathlib import Path

import inspect

def normalize_eval_strategy_kwargs(kwargs: dict) -> dict:
    """Normalize evaluation strategy kwarg across Transformers versions.

    Some environments accept `evaluation_strategy`, others accept `eval_strategy`.
    We convert to the name supported by the installed `TrainingArguments`.
    """
    params = inspect.signature(TrainingArguments.__init__).parameters
    out = dict(kwargs)
    if "evaluation_strategy" in params and "eval_strategy" in out:
        out["evaluation_strategy"] = out.pop("eval_strategy")
    elif "eval_strategy" in params and "evaluation_strategy" in out:
        out["eval_strategy"] = out.pop("evaluation_strategy")
    return out

def append_run_to_csv(csv_path: str, row: dict):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(row.keys())
    file_exists = csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# ================= 設定エリア =================
DATASET_DIR = "/home/mdxuser/datasets_and_models/kuzushiji_classifier_data_codh"
# MODEL_CHECKPOINT = "microsoft/swin-tiny-patch4-window7-224"
# 修正後（強くてニューゲーム）
MODEL_CHECKPOINT = "microsoft/swin-base-patch4-window7-224"
OUTPUT_DIR = "/home/mdxuser/datasets_and_models/kuzushiji_class_models/codh_only/min10"
dataset_name = Path(DATASET_DIR).name
tracking_csv = str(Path(OUTPUT_DIR) / "run_tracking.csv")
# Filter out rare classes before splitting (needed for stratified split + stable macro metrics)
MIN_COUNT = 10  # change to 10/20/50 depending on your experiment


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

    print(f"Loading dataset from CSV: {DATASET_DIR}/metadata.csv ...")
    csv_path = os.path.join(DATASET_DIR, "metadata.csv")

    # Force CSV schema: some IDs can be alphanumeric (e.g., "brsk00000"), so treat them as strings
    csv_features = Features({
        "id": Value("int64"),
        "file_name": Value("string"),
        "unicode": Value("string"),
        "char": Value("string"),
        "book_id": Value("string"),
        "page_image": Value("string"),
        "x": Value("int64"),
        "y": Value("int64"),
        "w": Value("int64"),
        "h": Value("int64"),
    })
    print("DEBUG: using csv_features schema for load_dataset")
    dataset = load_dataset("csv", data_files=csv_path, features=csv_features)

    # --- DRY RUN: shrink dataset early to avoid expensive full mapping ---
    if args_cli.dry_run:
        print("⚠️  [DRY RUN MODE] Pre-shrinking raw CSV rows")
        dataset["train"] = dataset["train"].select(
            range(min(5000, len(dataset["train"])))
        )

    # Add absolute path to each image and cast to Image so transforms get PIL Images
    def add_image_path(ex):
        # file_name is like "images/00000001.jpg"
        ex["image"] = os.path.join(DATASET_DIR, ex["file_name"])
        return ex

    dataset = dataset.map(add_image_path, num_proc=4)
    dataset = dataset.cast_column("image", Image())
    
    # --- Explicitly set label column from metadata ---
    LABEL_COLUMN = "char"   # character classification

    for split in dataset:
        if LABEL_COLUMN not in dataset[split].column_names:
            raise ValueError(f"Label column '{LABEL_COLUMN}' not found in metadata.csv")

    # ---- Build initial label vocab from char ----
    unique_labels = sorted(set(dataset["train"][LABEL_COLUMN]))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}

    def encode_labels(example):
        example["labels"] = label2id[example[LABEL_COLUMN]]
        return example

    dataset = dataset.map(encode_labels, num_proc=4)

    # ---- MIN_COUNT filtering (remove rare classes BEFORE splitting) ----
    if MIN_COUNT is not None and MIN_COUNT > 1:
        from collections import Counter
        from datasets import ClassLabel

        counts = Counter(dataset["train"]["labels"])
        keep_old_ids = sorted([lab for lab, c in counts.items() if c >= MIN_COUNT])
        print(f"🧹 MIN_COUNT={MIN_COUNT}: keeping {len(keep_old_ids)} / {len(counts)} classes")

        keep_set = set(keep_old_ids)

        # Filter using only the labels column (avoids forcing image decoding work)
        dataset["train"] = dataset["train"].filter(
            lambda lab: lab in keep_set,
            input_columns=["labels"],
            num_proc=4,
        )

        # Remap label ids to 0..K-1 (required after filtering)
        old2new = {old: new for new, old in enumerate(keep_old_ids)}

        def remap_label(example):
            example["labels"] = old2new[example["labels"]]
            return example

        dataset["train"] = dataset["train"].map(remap_label, num_proc=4)

        # Rebuild label name mappings to match the filtered/remapped ids
        kept_names = [id2label[old] for old in keep_old_ids]
        label2id = {name: i for i, name in enumerate(kept_names)}
        id2label = {i: name for i, name in enumerate(kept_names)}

        # Cast labels to ClassLabel so `stratify_by_column="labels"` works
        dataset["train"] = dataset["train"].cast_column("labels", ClassLabel(names=kept_names))

    # 【修正箇所】 検証用データがない場合、自動的に分割を作成する
    if "validation" not in dataset and "test" not in dataset:
        print("⚠️ No validation/test split found. Splitting train dataset (90% Train, 10% Test)...")
        # seedを固定して再現性を確保
        dataset = dataset["train"].train_test_split(
            test_size=0.1,
            seed=42,
            stratify_by_column="labels",
        )
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

    # クラス情報の取得（char->labels で作った辞書をそのまま使う）
    num_labels = len(id2label)
    print(f"✅ Found {num_labels} classes.")

    # 検証用スプリット名の確定
    val_split_name = "test" if "test" in dataset else "validation"

    # 前処理の適用
    train_ds = dataset["train"].with_transform(train_transforms)
    val_ds = dataset[val_split_name].with_transform(val_transforms)

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    training_args_config = normalize_eval_strategy_kwargs(training_args_config)
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
    
    eval_metrics = trainer.evaluate()

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": dataset_name,
        "dataset_dir": DATASET_DIR,
        "model_checkpoint": MODEL_CHECKPOINT,
        "output_dir": OUTPUT_DIR,
        "seed_split": 42,                  # your split seed
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "val_split": ("test" if "test" in dataset else "validation"),
        "train_size": len(dataset["train"]),
        "val_size": len(dataset["test"] if "test" in dataset else dataset["validation"]),
        "eval_accuracy": eval_metrics.get("eval_accuracy"),
        "eval_loss": eval_metrics.get("eval_loss"),
        "best_metric": getattr(trainer.state, "best_metric", None),
        "best_model_checkpoint": getattr(trainer.state, "best_model_checkpoint", None),
    }

    append_run_to_csv(tracking_csv, row)
    print(f"📝 Run logged to: {tracking_csv}")

    print("Saving model...")
    trainer.save_model()
    image_processor.save_pretrained(OUTPUT_DIR)
    print(f"\n✅ Training Completed. Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()