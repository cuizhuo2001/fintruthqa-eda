"""评估部分的公共代码：读数据、固定划分训练/测试集、算指标。

三个评估脚本（eval_tfidf.py / eval_bert.py / eval_llm.py）都从这里拿同一份划分，
保证所有模型是在同一批测试题上比的。
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
CLEAN_CSV = ROOT / "output" / "fintruthqa_clean.csv"
EVAL_DIR = ROOT / "output" / "eval"
PRED_DIR = EVAL_DIR / "predictions"
FIG_DIR = ROOT / "figures"

TARGET = "ans_relevance"
LABELS = [1, 2, 3]
SEED = 42
TEST_SIZE = 0.2


def load_data():
    return pd.read_csv(CLEAN_CSV, encoding="utf-8-sig")


def get_split():
    """80/20 分层划分。按「相关性 × 是否真问题」分层，两类少数样本在两边比例一样。

    划分结果写到 output/eval/split.csv；以后都读这份文件，不重新随机。
    """
    path = EVAL_DIR / "split.csv"
    df = load_data()
    if not path.exists():
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        strata = df[TARGET].astype(str) + "_" + df["is_question"].astype(str)
        train_ids, test_ids = train_test_split(df["id"], test_size=TEST_SIZE, stratify=strata, random_state=SEED)
        split = pd.DataFrame({"id": df["id"], "split": np.where(df["id"].isin(test_ids), "test", "train")})
        split.to_csv(path, index=False)
    split = pd.read_csv(path)
    df = df.merge(split, on="id")
    return df[df["split"] == "train"].reset_index(drop=True), df[df["split"] == "test"].reset_index(drop=True)


def pilot_ids(test, n=300):
    """从测试集里按相关性分层抽 n 条，给 LLM 试跑估费用用。"""
    ids, _ = train_test_split(test["id"], train_size=n, stratify=test[TARGET], random_state=SEED)
    return set(ids)


def save_predictions(name, ids, y_pred):
    """每个模型的预测存成 output/eval/predictions/<name>.csv（id, pred）。"""
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, "pred": y_pred}).to_csv(PRED_DIR / f"{name}.csv", index=False)


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return {
        "n": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=LABELS)),
        "kappa_quadratic": float(cohen_kappa_score(y_true, y_pred, labels=LABELS, weights="quadratic")),
        **{f"f1_class{c}": float(v) for c, v in zip(LABELS, per_class)},
    }


def bootstrap_ci(y_true, y_pred, n_boot=1000, seed=SEED):
    """macro-F1 和 Kappa 的 95% bootstrap 置信区间（测试集有放回重抽 1000 次）。"""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1s, kappas = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        f1s.append(f1_score(y_true[idx], y_pred[idx], labels=LABELS, average="macro", zero_division=0))
        kappas.append(cohen_kappa_score(y_true[idx], y_pred[idx], labels=LABELS))
    return {
        "macro_f1_ci": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
        "kappa_ci": [float(np.percentile(kappas, 2.5)), float(np.percentile(kappas, 97.5))],
    }


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
