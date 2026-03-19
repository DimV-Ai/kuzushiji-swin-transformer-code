#!/usr/bin/env python3
"""
Native evaluation suite for the 2 combined CODH+Ogihan stratified kuzushiji classifiers.

What it does
------------
For each combined model, it rebuilds the *native* evaluation split in the same way as the
corresponding training script:
- load the HF combined dataset
- attach `char` labels from metadata.csv using `__key__`
- rebuild label vocabulary from the dataset
- apply MIN_COUNT filtering before splitting
- split with seed=42 and test_size=0.1
- evaluate the saved model on its own native test split

If the rebuilt label space does not exactly match the saved model's label2id,
the script saves diagnostics and then evaluates on the overlapping label space.

Outputs
-------
- summary CSV with one row per model
- per-class metrics CSV per model
- confusion-pairs CSV per model
- macro-F1-vs-support-threshold CSV per model
- label-space diagnostics CSVs when a mismatch is detected

Usage
-----
Save this file as something like:
  native_eval_codhogihan.py

Then run:
  python native_eval_codhogihan.py

or with a custom output root:
  python native_eval_codhogihan.py --out_root /path/to/out
"""

from __future__ import annotations

import argparse
import re
import tarfile
import csv
import os
from collections import Counter
from pathlib import Path
from PIL import Image as PILImage
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import ClassLabel, Dataset, Features, Image, Value, load_dataset
from huggingface_hub import snapshot_download
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split as sk_train_test_split
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from transformers import AutoImageProcessor, AutoModelForImageClassification


# -----------------------------------------------------------------------------
# DEDICATED SPECS FOR COMBINED CODH+OGIHAN MODELS
# -----------------------------------------------------------------------------
MODEL_SPECS = [
    {
        "name": "codhogihan_min10",
        "dataset_kind": "codh_ogihan",
        "min_count": 10,
        "model_dir": "/home/mdxuser/datasets_and_models/kuzushiji_class_models/codh_and_ogihan/min10",
        "hf_dataset_id": "DimV-Ai/kuzushiji-character-dataset-v1",
    },
    {
        "name": "codhogihan_min20",
        "dataset_kind": "codh_ogihan",
        "min_count": 20,
        "model_dir": "/home/mdxuser/datasets_and_models/kuzushiji_class_models/codh_and_ogihan/min20",
        "hf_dataset_id": "DimV-Ai/kuzushiji-character-dataset-v1",
    },
]

SPLIT_SEED = 42
TEST_SIZE = 0.1
BATCH_SIZE = 128
THRESHOLDS = [1, 2, 5, 10, 20, 50, 100]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------------------------
# Dataset builders mirroring the training scripts
# -----------------------------------------------------------------------------
def extract_unicode_from_filename(filename: str) -> str:
    m = re.search(r"U\+[0-9A-Fa-f]+", filename)
    if not m:
        raise ValueError(f"Could not extract Unicode label from filename: {filename}")
    return m.group(0).upper()


def load_member_to_char_map(metadata_csv_path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with open(metadata_csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"member", "char"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"metadata.csv missing columns: {sorted(missing)}")
        for row in reader:
            member = (row["member"] or "").strip()
            char = (row["char"] or "").strip()
            if member and char:
                mapping[member] = char
    if not mapping:
        raise ValueError("No member->char mappings were loaded")
    return mapping


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


def build_ogihan_dataset() -> Dataset:
    # Mirror the actual Ogihan training scripts:
    # snapshot the repo, extract tarred JPG crops, then build labels from filenames.
    repo_dir = Path(snapshot_download(repo_id="DimV-Ai/kuzushiji-character-dataset-ogihan-v1", repo_type="dataset"))
    extract_root = repo_dir / "_extracted_images"
    extract_tars_if_needed(repo_dir, extract_root)
    ds = build_dataset_from_extracted_images(extract_root)
    return ds


def build_codh_only_dataset(dataset_dir: str) -> Dataset:
    csv_path = os.path.join(dataset_dir, "metadata.csv")
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
    ds_dict = load_dataset("csv", data_files=csv_path, features=csv_features)
    ds = ds_dict["train"]

    def add_image_path(ex):
        ex["image"] = os.path.join(dataset_dir, ex["file_name"])
        return ex

    ds = ds.map(add_image_path)
    ds = ds.cast_column("image", Image())
    return ds


def build_codh_ogihan_dataset(hf_dataset_id: str) -> Dataset:
    ds_dict = load_dataset(hf_dataset_id)
    ds = ds_dict["train"]

    cols = ds.column_names
    print("HF dataset columns:", cols)

    if "jpg" in cols:
        ds = ds.rename_column("jpg", "image")
        ds = ds.cast_column("image", Image())
    elif "image" in cols:
        ds = ds.cast_column("image", Image())
    else:
        raise ValueError(f"Expected an image column ('jpg' or 'image'). Available columns: {cols}")

    repo_dir = snapshot_download(repo_id=hf_dataset_id, repo_type="dataset")
    metadata_csv_path = os.path.join(repo_dir, "metadata.csv")
    member_to_char = load_member_to_char_map(metadata_csv_path)

    def attach_char(example):
        if "__key__" not in example:
            raise KeyError(f"Expected '__key__' in dataset example. Available keys: {list(example.keys())}")
        member = f"{example['__key__']}.jpg"
        char = member_to_char.get(member)
        if char is None:
            raise KeyError(f"No char mapping for image member: {member}")
        example["char"] = char
        return example

    ds = ds.map(attach_char, desc="Attaching char labels from metadata.csv")
    return ds


def prepare_native_split(spec: dict) -> Tuple[Dataset, Dict[str, int], Dict[int, str], str]:
    kind = spec["dataset_kind"]
    min_count = int(spec["min_count"])

    if kind == "codh_ogihan":
        ds = build_codh_ogihan_dataset(spec["hf_dataset_id"])
        label_column = "char"
    else:
        raise ValueError(f"Unknown dataset_kind for this dedicated script: {kind}")

    # Build the label vocabulary from the attached character labels.
    raw_labels = ds[label_column]
    unique_labels = sorted(set(raw_labels))
    raw_label2id = {label: i for i, label in enumerate(unique_labels)}
    raw_ids = np.array([raw_label2id[label] for label in raw_labels], dtype=np.int32)

    counts = Counter(raw_ids.tolist())
    keep_old_ids = sorted([lab for lab, c in counts.items() if c >= min_count])
    keep_old_ids_set = set(keep_old_ids)

    kept_indices = np.array([i for i, lab in enumerate(raw_ids) if lab in keep_old_ids_set], dtype=np.int64)
    kept_old_ids = raw_ids[kept_indices]

    old_id_to_new_id = {old: new for new, old in enumerate(keep_old_ids)}
    remapped_ids = np.array([old_id_to_new_id[lab] for lab in kept_old_ids], dtype=np.int32)

    kept_names = [unique_labels[old] for old in keep_old_ids]
    label2id = {name: i for i, name in enumerate(kept_names)}
    id2label = {i: name for i, name in enumerate(kept_names)}

    # Split on indices only, so we avoid materializing multiple huge Arrow caches on disk.
    train_idx, test_idx = sk_train_test_split(
        kept_indices,
        test_size=TEST_SIZE,
        random_state=SPLIT_SEED,
        stratify=remapped_ids,
    )

    test_idx = np.array(sorted(test_idx.tolist()), dtype=np.int64)
    eval_ds = ds.select(test_idx.tolist())

    eval_label_values = [int(label2id[ch]) for ch in eval_ds[label_column]]
    eval_ds = eval_ds.add_column("labels", eval_label_values)
    eval_ds = eval_ds.cast_column("labels", ClassLabel(num_classes=len(kept_names), names=kept_names))

    return eval_ds, label2id, id2label, label_column


# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------
def make_transform(processor):
    normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
    size = processor.size["shortest_edge"] if "shortest_edge" in processor.size else (processor.size["height"], processor.size["width"])
    return Compose([Resize(size), ToTensor(), normalize])

def ensure_pil_image(img_obj):
    if isinstance(img_obj, PILImage.Image):
        return img_obj
    if isinstance(img_obj, str):
        return PILImage.open(img_obj)
    if isinstance(img_obj, dict):
        if "path" in img_obj and img_obj["path"]:
            return PILImage.open(img_obj["path"])
        if "bytes" in img_obj and img_obj["bytes"] is not None:
            import io
            return PILImage.open(io.BytesIO(img_obj["bytes"]))
    raise TypeError(f"Unsupported image object type: {type(img_obj)}")


# Helper to resolve model dir
def resolve_model_dir(model_dir: str) -> str:
    p = Path(model_dir)

    if not p.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_dir}")
    if not p.is_dir():
        raise NotADirectoryError(f"Model path is not a directory: {model_dir}")

    direct_files = {x.name for x in p.iterdir() if x.is_file()}
    if "config.json" in direct_files and (
        "preprocessor_config.json" in direct_files or "image_processor_config.json" in direct_files
    ):
        return str(p)

    candidates = []
    for child in sorted(p.iterdir()):
        if not child.is_dir():
            continue
        child_files = {x.name for x in child.iterdir() if x.is_file()}
        if "config.json" in child_files and (
            "preprocessor_config.json" in child_files or "image_processor_config.json" in child_files
        ):
            candidates.append(child)

    if len(candidates) == 1:
        print(f"Resolved model directory for {model_dir} -> {candidates[0]}")
        return str(candidates[0])
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple possible HF model directories found under {model_dir}: "
            f"{[str(c) for c in candidates]}"
        )

    raise FileNotFoundError(
        f"Could not find a Hugging Face model directory at {model_dir}. "
        f"Looked for config.json plus preprocessor_config.json/image_processor_config.json "
        f"in that directory and one level below."
    )


@torch.no_grad()
def run_model(model, processor, ds: Dataset, batch_size: int = BATCH_SIZE):
    tfm = make_transform(processor)
    model.eval().to(DEVICE)

    all_logits = []
    all_labels = []

    for start in range(0, len(ds), batch_size):
        batch = ds.select(range(start, min(start + batch_size, len(ds))))
        images = [tfm(ensure_pil_image(img).convert("RGB")) for img in batch["image"]]
        pixel_values = torch.stack(images).to(DEVICE)
        labels = np.array(batch["labels"])
        logits = model(pixel_values=pixel_values).logits.detach().cpu().numpy()
        all_logits.append(logits)
        all_labels.append(labels)

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    preds = np.argmax(logits, axis=1)
    return logits, labels, preds


def compute_global_metrics(logits, labels, preds) -> dict:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    top5 = np.argsort(-logits, axis=1)[:, :5]
    top5_acc = float(np.mean([lab in row for lab, row in zip(labels, top5)]))
    acc = float(np.mean(preds == labels))
    total_errors = int(np.sum(preds != labels))
    return {
        "n_samples": int(len(labels)),
        "n_classes": int(len(np.unique(labels))),
        "accuracy": acc,
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "top5_accuracy": top5_acc,
        "total_errors": total_errors,
    }


def save_per_class(eval_dir: str, labels, preds, id2label: Dict[int, str]) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    rows = []
    for class_id in range(len(precision)):
        rows.append({
            "char": id2label[class_id],
            "support": int(support[class_id]),
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "f1": float(f1[class_id]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(eval_dir, "per_class_metrics.csv"), index=False, encoding="utf-8")
    return df


def save_confusions(eval_dir: str, labels, preds, id2label: Dict[int, str]) -> pd.DataFrame:
    cm = confusion_matrix(labels, preds)
    rows = []
    for true_id in range(cm.shape[0]):
        for pred_id in range(cm.shape[1]):
            count = int(cm[true_id, pred_id])
            if true_id != pred_id and count > 0:
                rows.append({
                    "true_char": id2label[true_id],
                    "pred_char": id2label[pred_id],
                    "count": count,
                })
    df = pd.DataFrame(rows).sort_values("count", ascending=False)
    df.to_csv(os.path.join(eval_dir, "confusion_pairs.csv"), index=False, encoding="utf-8")
    return df


def save_threshold_metrics(eval_dir: str, per_class_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in THRESHOLDS:
        sub = per_class_df[per_class_df["support"] >= t]
        rows.append({
            "support_threshold": t,
            "n_classes": int(len(sub)),
            "macro_precision": float(sub["precision"].mean()) if len(sub) else np.nan,
            "macro_recall": float(sub["recall"].mean()) if len(sub) else np.nan,
            "macro_f1": float(sub["f1"].mean()) if len(sub) else np.nan,
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(eval_dir, "macro_metrics_by_support_threshold.csv"), index=False, encoding="utf-8")
    return df


# ---------------------------------------------------------------------------
# Label-space diagnostics helper
# ---------------------------------------------------------------------------

def save_label_space_diagnostics(
    eval_dir: str,
    name: str,
    eval_ds: Dataset,
    label_column: str,
    rebuilt_label2id: Dict[str, int],
    model_label2id: Dict[str, int],
) -> None:
    rebuilt_counts = Counter(eval_ds[label_column])
    rebuilt_rows = []
    for label in sorted(rebuilt_label2id.keys()):
        rebuilt_rows.append(
            {
                "label": label,
                "rebuilt_id": int(rebuilt_label2id[label]),
                "rebuilt_eval_count": int(rebuilt_counts.get(label, 0)),
            }
        )
    pd.DataFrame(rebuilt_rows).to_csv(
        os.path.join(eval_dir, "rebuilt_label_space.csv"),
        index=False,
        encoding="utf-8",
    )

    model_rows = []
    for label, idx in sorted(model_label2id.items(), key=lambda kv: int(kv[1])):
        model_rows.append(
            {
                "label": label,
                "model_id": int(idx),
            }
        )
    pd.DataFrame(model_rows).to_csv(
        os.path.join(eval_dir, "model_label_space.csv"),
        index=False,
        encoding="utf-8",
    )

    rebuilt_keys = set(rebuilt_label2id.keys())
    model_keys = set(model_label2id.keys())
    only_rebuilt = sorted(rebuilt_keys - model_keys)
    only_model = sorted(model_keys - rebuilt_keys)

    diff_rows = []
    for label in only_rebuilt:
        diff_rows.append(
            {
                "label": label,
                "status": "rebuilt_only",
                "rebuilt_eval_count": int(rebuilt_counts.get(label, 0)),
            }
        )
    for label in only_model:
        diff_rows.append(
            {
                "label": label,
                "status": "model_only",
                "rebuilt_eval_count": int(rebuilt_counts.get(label, 0)),
            }
        )
    pd.DataFrame(diff_rows).to_csv(
        os.path.join(eval_dir, "label_space_diff.csv"),
        index=False,
        encoding="utf-8",
    )

    print(f"Saved label-space diagnostics for {name} to: {eval_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=str, default="./native_eval_outputs")
    args = parser.parse_args()

    out_root = args.out_root
    os.makedirs(out_root, exist_ok=True)

    summary_rows = []

    for spec in MODEL_SPECS:
        name = spec["name"]
        print(f"\n===== Native evaluation: {name} =====")
        eval_dir = os.path.join(out_root, name)
        os.makedirs(eval_dir, exist_ok=True)

        eval_ds, rebuilt_label2id, rebuilt_id2label, label_column = prepare_native_split(spec)

        resolved_model_dir = resolve_model_dir(spec["model_dir"])
        processor = AutoImageProcessor.from_pretrained(resolved_model_dir, local_files_only=True)
        model = AutoModelForImageClassification.from_pretrained(resolved_model_dir, local_files_only=True)

        model_label2id = getattr(model.config, "label2id", None) or {}
        model_label_keys = set(model_label2id.keys())
        rebuilt_keys = set(rebuilt_label2id.keys())

        missing_in_model = sorted(rebuilt_keys - model_label_keys)
        missing_in_rebuilt = sorted(model_label_keys - rebuilt_keys)

        if model_label_keys != rebuilt_keys:
            save_label_space_diagnostics(
                eval_dir=eval_dir,
                name=name,
                eval_ds=eval_ds,
                label_column=label_column,
                rebuilt_label2id=rebuilt_label2id,
                model_label2id=model_label2id,
            )
            print(f"WARNING: Label-space mismatch for {name}")
            print(f"Rebuilt label count: {len(rebuilt_keys)}")
            print(f"Model label count:   {len(model_label_keys)}")
            print(f"Sample rebuilt-only labels: {missing_in_model[:20]}")
            print(f"Sample model-only labels:   {missing_in_rebuilt[:20]}")

            common_labels = sorted(rebuilt_keys & model_label_keys)
            if not common_labels:
                raise ValueError(f"No overlapping labels between rebuilt split and model for {name}")

            common_label_set = set(common_labels)
            before_n = len(eval_ds)
            eval_ds = eval_ds.filter(
                lambda lab: lab in common_label_set,
                input_columns=[label_column],
                desc="Filtering eval split to common label space",
            )
            after_n = len(eval_ds)
            print(
                f"Filtered eval split to common labels: kept {after_n}/{before_n} rows "
                f"({after_n / before_n:.2%} retained)"
            )

        # Align native split labels to the saved model's label IDs.
        def relabel_for_model(ex):
            ex["labels"] = int(model_label2id[ex[label_column]])
            return ex

        eval_ds = eval_ds.map(relabel_for_model, desc="Relabeling eval split to model label IDs")
        id2label = {int(v): k for k, v in model_label2id.items()}

        logits, labels, preds = run_model(model, processor, eval_ds)
        metrics = compute_global_metrics(logits, labels, preds)
        per_class_df = save_per_class(eval_dir, labels, preds, id2label)
        save_confusions(eval_dir, labels, preds, id2label)
        save_threshold_metrics(eval_dir, per_class_df)

        metrics.update({
            "model": name,
            "dataset_kind": spec["dataset_kind"],
            "min_count": spec["min_count"],
            "model_dir": resolved_model_dir,
            "n_model_only_labels": len(missing_in_rebuilt),
            "n_rebuilt_only_labels": len(missing_in_model),
        })
        summary_rows.append(metrics)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df[[
        "model", "dataset_kind", "min_count", "n_samples", "n_classes",
        "accuracy", "precision_macro", "recall_macro", "f1_macro",
        "precision_weighted", "recall_weighted", "f1_weighted",
        "top5_accuracy", "total_errors",
        "n_model_only_labels", "n_rebuilt_only_labels",
        "model_dir",
    ]]
    summary_path = os.path.join(out_root, "native_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"\nSaved native summary to: {summary_path}")
    print(summary_df.sort_values(["min_count", "f1_macro"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
