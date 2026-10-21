"""LLM 打分：用 DeepSeek 给测试集的回答相关性打 1/2/3 分，零样本和少样本各跑一遍。

python -I eval_llm.py --pilot      先在测试集里分层抽 300 条试跑，看花多少钱
python -I eval_llm.py              跑全部测试集（1,200 条），试跑过的直接用缓存

- API key：先读环境变量 DEEPSEEK_API_KEY，没有就读本机 Codex 配置
  （D:\\codex\\home\\config.toml 里 deepseek 那把）。key 不会写进仓库。
- 每条的模型原始输出缓存在 output/eval/llm_cache/<模式>.jsonl，重跑只补没跑过的，不重复花钱。
- 提示词里把「提问是不是真问题」作为已知条件给模型（原因见 README「主要发现 4」）。
- 少样本的示例从训练集里固定抽 8 条，测试集里的题不会出现在示例里。
"""

import argparse
import json
import os
import re
import sys
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I 不会自动把脚本目录加进来
from eval_common import EVAL_DIR, SEED, TARGET, get_split, metrics, pilot_ids, save_predictions, write_json  # noqa: E402

API_BASE = "https://api.deepseek.com"
MODEL = "deepseek-flash"  # /models 返回的名字是 DeepSeek-V4.1-Flash
CODEX_CONFIG = Path(r"D:\codex\home\config.toml")
CACHE_DIR = EVAL_DIR / "llm_cache"
WORKERS = 8

SYSTEM_PROMPT = """你是上市公司信息披露质量的标注员。任务：看投资者在交易所互动平台上的提问和公司的回答，按官方标注指南给「回答相关性」打 1、2 或 3 分。

评分标准：
1 = 完全不相关：提问里的问题一个也没有正面回答。只说客套话（如「感谢关注」）、打官腔（如「股价受多种因素影响」「公司会做好经营」）、或者说了相关的话题但没有回答到点上，都算 1。
2 = 部分相关或难以判断：只正面回答了一部分问题，或者很难判断有没有正面回答。
3 = 完全相关：正面回答了提问里的全部问题（一句话同时回答了几个问题也算）。如果提问说的事和公司无关，公司明确否定即可，不用回答后面的问题，也算 3。

另外会告诉你这条提问是不是一个真问题（这是标注员事先判断好的，直接采用）：
- 是真问题：按上面的标准，看公司有没有正面回答。
- 不是真问题（只是建议、评论、祝福或抱怨）：对建议、评论、祝福，公司表示收到、感谢、会转达或会考虑，一般算 3；对抱怨和质疑，公司要正面回应抱怨的内容才算相关，只说客套话算 1。

只输出一个数字（1、2 或 3），不要输出任何其他内容。"""


def user_message(row):
    is_q = "是" if row["is_question"] == 1 else "不是（建议/评论/祝福/抱怨）"
    return f"提问是不是真问题：{is_q}\n提问：{row['question']}\n回答：{row['answer']}"


def few_shot_examples(train):
    """从训练集抽 8 条短一点的示例：真问题里 1/2/3 分各 2 条，非问题里 3 分和 1 分各 1 条。"""
    short = train[(train["q_len"] <= 80) & (train["a_len"] <= 120) & ~train["flag_label_rule_conflict"]]
    picks = []
    for is_q, label, n in [(1, 1, 2), (1, 2, 2), (1, 3, 2), (0, 3, 1), (0, 1, 1)]:
        pool = short[(short["is_question"] == is_q) & (short[TARGET] == label)]
        picks.append(pool.sample(n, random_state=SEED))
    # 打乱顺序，别让模型学到「最后一个示例是几分」
    return pd.concat(picks).sample(frac=1, random_state=SEED)


def build_messages(row, shots):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if shots is not None:
        for _, ex in shots.iterrows():
            messages.append({"role": "user", "content": user_message(ex)})
            messages.append({"role": "assistant", "content": str(ex[TARGET])})
    messages.append({"role": "user", "content": user_message(row)})
    return messages


def load_api_key():
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    if CODEX_CONFIG.exists():
        cfg = tomllib.loads(CODEX_CONFIG.read_text(encoding="utf-8"))
        key = cfg.get("model_providers", {}).get("deepseek", {}).get("experimental_bearer_token")
        if key:
            return key
    sys.exit("找不到 DeepSeek API key：请设置环境变量 DEEPSEEK_API_KEY")


def get_balance(key):
    try:
        r = requests.get(f"{API_BASE}/user/balance", headers={"Authorization": f"Bearer {key}"}, timeout=30)
        return float(r.json()["balance_infos"][0]["total_balance"])
    except Exception:
        return None


def call_llm(key, messages):
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 5,
        "thinking": {"type": "disabled"},  # 不要思考过程，只要一个数字
    }
    for attempt in range(5):
        try:
            r = requests.post(f"{API_BASE}/chat/completions", headers={"Authorization": f"Bearer {key}"},
                              json=body, timeout=60)
            if r.status_code == 200:
                j = r.json()
                return j["choices"][0]["message"]["content"], j["usage"]
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except requests.RequestException:
            time.sleep(2 ** attempt)
    raise RuntimeError("连续 5 次请求失败")


def parse_label(raw):
    m = re.search(r"[123]", raw or "")
    return int(m.group()) if m else None


def load_cache(mode):
    path = CACHE_DIR / f"{mode}.jsonl"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {rec["id"]: rec for rec in map(json.loads, f)}


def run_mode(key, mode, rows, shots):
    cache = load_cache(mode)
    todo = [r for _, r in rows.iterrows() if int(r["id"]) not in cache]
    print(f"[{mode}] 共 {len(rows)} 条，缓存里已有 {len(rows) - len(todo)} 条，这次要调 {len(todo)} 次 API")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    with (CACHE_DIR / f"{mode}.jsonl").open("a", encoding="utf-8") as f, ThreadPoolExecutor(WORKERS) as pool:
        futures = {pool.submit(call_llm, key, build_messages(r, shots)): int(r["id"]) for r in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            raw, usage = fut.result()
            rec = {"id": futures[fut], "raw": raw, "pred": parse_label(raw),
                   "prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"],
                   "cache_hit_tokens": usage.get("prompt_cache_hit_tokens", 0)}
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                cache[rec["id"]] = rec
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}")
    return [cache[int(i)] for i in rows["id"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="只跑测试集里分层抽的 300 条")
    args = ap.parse_args()

    train, test = get_split()
    rows = test[test["id"].isin(pilot_ids(test))] if args.pilot else test
    shots = few_shot_examples(train)
    key = load_api_key()

    balance_before = get_balance(key)
    usage_summary = {}
    for mode, mode_shots in [("zero_shot", None), ("few_shot", shots)]:
        recs = run_mode(key, mode, rows, mode_shots)
        bad = [r["id"] for r in recs if r["pred"] is None]
        if bad:
            print(f"  [{mode}] {len(bad)} 条输出解析不出 1/2/3，按 2（难以判断）处理：{bad[:10]}")
        preds = [r["pred"] or 2 for r in recs]
        if not args.pilot:
            save_predictions(f"llm_{mode}", rows["id"], preds)
        usage_summary[mode] = {
            "n": len(recs),
            "unparsed": len(bad),
            "prompt_tokens": sum(r["prompt_tokens"] for r in recs),
            "cache_hit_tokens": sum(r["cache_hit_tokens"] for r in recs),
            "completion_tokens": sum(r["completion_tokens"] for r in recs),
        }
    time.sleep(5)  # 余额扣费有一点延迟
    balance_after = get_balance(key)

    if args.pilot:
        # 试跑结果单独存，不覆盖正式预测
        y = rows.set_index("id")[TARGET]
        for mode in ["zero_shot", "few_shot"]:
            cache = load_cache(mode)
            m = metrics(y.values, [cache[int(i)]["pred"] or 2 for i in y.index])
            print(f"  试跑 [{mode}] macro-F1={m['macro_f1']:.3f}  Kappa={m['kappa']:.3f}")

    spent = None if balance_before is None or balance_after is None else round(balance_before - balance_after, 4)
    print(f"余额 {balance_before} → {balance_after} 元，这次花了约 {spent} 元")
    write_json(EVAL_DIR / f"llm_run_{'pilot' if args.pilot else 'full'}.json", {
        "model": MODEL,
        "n_rows": len(rows),
        "few_shot_ids": [int(i) for i in shots["id"]],
        "usage": usage_summary,
        "balance_before_cny": balance_before,
        "balance_after_cny": balance_after,
        "spent_this_run_cny": spent,
    })


if __name__ == "__main__":
    main()
