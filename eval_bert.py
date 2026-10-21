"""基线 2：中文 BERT（bert-base-chinese）微调，预测回答相关性（1/2/3）。

python -I eval_bert.py

- 输入：「[真问题：是/不是] + 问题」和「回答」拼成一对句子，最长 256 个 token（超过的只占 1–2%）。
- 训练集再分出 10% 当验证集，按验证集 macro-F1 挑最好的一轮，再拿它预测测试集。
- 类别不平衡：损失函数按类别频率反比加权（和 TF-IDF 的 class_weight="balanced" 同一个思路）。
- 需要 torch + transformers；本机 GTX 1660 Ti 6G 实测每轮约 20 分钟，4 轮共约 85 分钟。
- 模型权重下载到 D:\\tools\\hf-cache（不进仓库）。GPU 上训练不是严格可复现的，重跑数字可能差零点几个百分点。
"""

import copy
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", r"D:\tools\hf-cache")
# anaconda 的 numpy(MKL) 和 torch 各带一份 OpenMP 运行库，Windows 上不设这个一 import torch 就退出
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I 不会自动把脚本目录加进来
from eval_common import EVAL_DIR, LABELS, SEED, TARGET, get_split, metrics, save_predictions, write_json  # noqa: E402

MODEL_NAME = "bert-base-chinese"
MAX_LEN = 256
BATCH = 16
EPOCHS = 4
LR = 2e-5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode(tok, df):
    first = [("[真问题：是]" if q == 1 else "[真问题：不是]") + text for q, text in zip(df["is_question"], df["question"])]
    enc = tok(first, list(df["answer"]), truncation="longest_first", max_length=MAX_LEN, padding=False)
    labels = (df[TARGET] - 1).tolist()  # 1/2/3 → 0/1/2
    return [{"input_ids": i, "token_type_ids": t, "attention_mask": a, "label": y}
            for i, t, a, y in zip(enc["input_ids"], enc["token_type_ids"], enc["attention_mask"], labels)]


def batches(items, shuffle, tok):
    order = list(range(len(items)))
    if shuffle:
        random.shuffle(order)
    for s in range(0, len(order), BATCH):
        chunk = [items[i] for i in order[s:s + BATCH]]
        feats = tok.pad([{k: v for k, v in c.items() if k != "label"} for c in chunk], return_tensors="pt")
        feats["labels"] = torch.tensor([c["label"] for c in chunk])
        yield feats


@torch.no_grad()
def predict(model, items, tok, device):
    model.eval()
    out = []
    for b in batches(items, False, tok):
        b.pop("labels")
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            logits = model(**{k: v.to(device) for k, v in b.items()}).logits
        out.extend(logits.argmax(-1).cpu().tolist())
    return np.array(out) + 1  # 0/1/2 → 1/2/3


def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备：{device}" + (f"（{torch.cuda.get_device_name(0)}）" if device == "cuda" else "，没有 GPU 会非常慢"))

    train_all, test = get_split()
    train, val = train_test_split(train_all, test_size=0.1, stratify=train_all[TARGET], random_state=SEED)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tr_items, va_items, te_items = encode(tok, train), encode(tok, val), encode(tok, test)
    print(f"训练 {len(tr_items)} / 验证 {len(va_items)} / 测试 {len(te_items)} 条")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3).to(device)
    weights = compute_class_weight("balanced", classes=np.array(LABELS), y=train[TARGET])
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * ((len(tr_items) + BATCH - 1) // BATCH)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    best_f1, best_state, history = -1.0, None, []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0, total = time.time(), 0.0
        for b in batches(tr_items, True, tok):
            b = {k: v.to(device) for k, v in b.items()}
            labels = b.pop("labels")
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(**b).logits
            loss = loss_fn(logits.float(), labels)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        val_pred = predict(model, va_items, tok, device)
        val_f1 = f1_score(val[TARGET], val_pred, labels=LABELS, average="macro")
        history.append({"epoch": epoch, "train_loss": total / (steps / EPOCHS), "val_macro_f1": float(val_f1)})
        print(f"第 {epoch} 轮：训练损失 {history[-1]['train_loss']:.3f}，验证 macro-F1 {val_f1:.3f}，用时 {time.time() - t0:.0f} 秒")
        if val_f1 > best_f1:
            best_f1, best_state = val_f1, copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})

    model.load_state_dict(best_state)
    pred = predict(model, te_items, tok, device)
    save_predictions("bert", test["id"], pred)
    m = metrics(test[TARGET], pred)
    print(f"测试集 macro-F1={m['macro_f1']:.3f}  Kappa={m['kappa']:.3f}  准确率={m['accuracy']:.3f}")
    write_json(EVAL_DIR / "bert_meta.json", {
        "model": MODEL_NAME, "max_len": MAX_LEN, "batch": BATCH, "epochs": EPOCHS, "lr": LR,
        "best_val_macro_f1": float(best_f1), "history": history,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
    })


if __name__ == "__main__":
    main()
