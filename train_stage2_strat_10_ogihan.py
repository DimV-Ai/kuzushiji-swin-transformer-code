import csv
import re
import tarfile
import argparse
import inspect
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import evaluate
from datasets import Dataset, Image, ClassLabel
from huggingface_hub import snapshot_download
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)
from torchvision.transforms import (
    Compose,
    Normalize,
    Resize,
    ToTensor,
    ColorJitter,
    RandomAffine,
)


def normalize_eval_strategy_kwargs(kwargs: dict) -> dict:
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


def extract_unicode_from_filename(filename: str) -> str:
    m = re.search(r"U\+[0-9A-Fa-f]+", filename)
    if not m:
        raise ValueError(f"Could not extract Unicode label from filename: {filename}")
    return m.group(0).upper()


def extract_tars_if_needed(repo_dir: Path, extract_root: Path):
    extract_root.mkdir(parents=True, exist_ok=True)

    if any(extract_root.rglob("*.jpg")):
        print(f"Using previously extracted images from: {extract_root}")
        return

    tar_paths = sorted((repo_dir / "tars").glob("*.tar.gz"))
    if not tar_paths:
        raise FileNotFoundError(f"No .tar.gz files found under: {repo_dir / 'tars'}")

    for tar_path in tar_paths:
        print(f"Extracting: {tar_path.name}")
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(path=extract_root)


def read_metadata_csvs(repo_dir: Path) -> int:
    metadata_dir = repo_dir / "metadata"
    csv_paths = sorted(metadata_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No metadata CSVs found under: {metadata_dir}")

    total_rows = 0
    for csv_path in csv_paths:
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"Unicode", "Image", "X", "Y", "Width", "Height", "Block ID", "Char ID"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"Metadata file {csv_path.name} is missing columns: {sorted(missing)}. "
                    f"Found columns: {reader.fieldnames}"
                )
            total_rows += sum(1 for _ in reader)

    return total_rows


def build_dataset_from_extracted_images(extract_root: Path) -> Dataset:
    image_paths = sorted(extract_root.rglob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No extracted JPG images found under: {extract_root}")

    records = []
    for img_path in image_paths:
        unicode_label = extract_unicode_from_filename(img_path.name)
        records.append(
            {
                "image": str(img_path),
                "unicode": unicode_label,
                "file_name": img_path.name,
            }
        )

    ds = Dataset.from_list(records)
    ds = ds.cast_column("image", Image())
    return ds


HF_DATASET_ID = "DimV-Ai/kuzushiji-character-dataset-ogihan-v1"
MODEL_CHECKPOINT = "microsoft/swin-base-patch4-window7-224"
OUTPUT_DIR = "/home/mdxuser/datasets_and_models/kuzushiji_class_models/ogihan_only/min10"
dataset_name = "kuzushiji-character-dataset-ogihan-v1"
tracking_csv = str(Path(OUTPUT_DIR) / "run_tracking.csv")
MIN_COUNT = 10

BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 5e-5


image_processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT, use_fast=False)
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)
accuracy_metric = evaluate.load("accuracy")


def train_transforms(examples):
    transforms = Compose([
        Resize(size),
        RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ColorJitter(brightness=0.2, contrast=0.2),
        ToTensor(),
        normalize,
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
    parser = argparse.ArgumentParser(description="Kuzushiji Classifier Training (Ogihan-only HF dataset)")
    parser.add_argument("--dry-run", action="store_true", help="Enable dry run mode (fast check)")
    return parser.parse_args()


def main():
    args_cli = parse_args()

    print(f"Downloading Ogihan-only dataset repo from HuggingFace: {HF_DATASET_ID} ...")
    repo_dir = Path(snapshot_download(repo_id=HF_DATASET_ID, repo_type="dataset"))
    print(f"Local HF snapshot: {repo_dir}")

    metadata_rows = read_metadata_csvs(repo_dir)
    print(f"Read {metadata_rows} rows from metadata CSV files")

    extract_root = repo_dir / "_extracted_images"
    extract_tars_if_needed(repo_dir, extract_root)

    print("Building image dataset from extracted JPG crops ...")
    train_dataset = build_dataset_from_extracted_images(extract_root)
    print(f"Extracted crop rows: {len(train_dataset)}")

    dataset = {"train": train_dataset}

    if args_cli.dry_run:
        print("⚠️  [DRY RUN MODE] Pre-shrinking raw Ogihan image rows")
        dataset["train"] = dataset["train"].select(range(min(5000, len(dataset["train"]))))

    LABEL_COLUMN = "unicode"
    unique_labels = sorted(set(dataset["train"][LABEL_COLUMN]))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}

    def encode_labels(example):
        example["labels"] = label2id[example[LABEL_COLUMN]]
        return example

    dataset["train"] = dataset["train"].map(encode_labels, num_proc=4)

    if MIN_COUNT is not None and MIN_COUNT > 1:
        counts = Counter(dataset["train"]["labels"])
        keep_old_ids = sorted([lab for lab, c in counts.items() if c >= MIN_COUNT])
        print(f"🧹 MIN_COUNT={MIN_COUNT}: keeping {len(keep_old_ids)} / {len(counts)} classes")

        keep_set = set(keep_old_ids)
        dataset["train"] = dataset["train"].filter(
            lambda lab: lab in keep_set,
            input_columns=["labels"],
            num_proc=4,
        )

        old2new = {old: new for new, old in enumerate(keep_old_ids)}

        def remap_label(example):
            example["labels"] = old2new[example["labels"]]
            return example

        dataset["train"] = dataset["train"].map(remap_label, num_proc=4)

        kept_names = [id2label[old] for old in keep_old_ids]
        label2id = {name: i for i, name in enumerate(kept_names)}
        id2label = {i: name for i, name in enumerate(kept_names)}
        dataset["train"] = dataset["train"].cast_column("labels", ClassLabel(names=kept_names))

    print("⚠️ No validation/test split found. Splitting train dataset (90% Train, 10% Test)...")
    dataset = dataset["train"].train_test_split(
        test_size=0.1,
        seed=42,
        stratify_by_column="labels",
    )

    if args_cli.dry_run:
        print("\n⚠️  [DRY RUN MODE] ENABLED ⚠️")
        print("   - 縮小後データで学習パイプラインを確認します")
        dataset["train"] = dataset["train"].select(range(min(200, len(dataset["train"]))))
        dataset["test"] = dataset["test"].select(range(min(50, len(dataset["test"]))))
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

    num_labels = len(id2label)
    print(f"✅ Found {num_labels} classes.")

    train_ds = dataset["train"].with_transform(train_transforms)
    val_ds = dataset["test"].with_transform(val_transforms)

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
        **training_args_config,
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
        "dataset_dir": HF_DATASET_ID,
        "model_checkpoint": MODEL_CHECKPOINT,
        "output_dir": OUTPUT_DIR,
        "seed_split": 42,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "min_count": MIN_COUNT,
        "label_column": LABEL_COLUMN,
        "train_size": len(dataset["train"]),
        "val_size": len(dataset["test"]),
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
    print(f"\n✅ Ogihan-only training completed. Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()