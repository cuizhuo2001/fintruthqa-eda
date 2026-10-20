"""FinTruthQA 数据整理：下载 → 清洗 → 描述性统计 → 出图。

一键重跑：  python -I fintruthqa_pipeline.py   （或双击 run.bat）

产出：
  data/raw/                 原始数据（固定到上游某个 commit，校验 sha256）
  output/fintruthqa_clean.csv   清洗后数据（utf-8-sig，Excel 直接打开不乱码）
  output/stats_report.md    全部统计表（脚本自动生成）
  output/summary.json       关键数字（README 里引用的数都来自这里）
  figures/*.png             3 张图
"""

import hashlib
import html
import json
import re
import shutil
import sys
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "output"
FIG_DIR = ROOT / "figures"

# 上游仓库 bethxx99/FinTruthQA，固定到 2026-05-07 的 commit，保证别人重跑拿到同一份数据
UPSTREAM_COMMIT = "4fc4d7a5a6a355ba89a82cbe24c6e95f41bead4b"
UPSTREAM_RAW = f"https://raw.githubusercontent.com/bethxx99/FinTruthQA/{UPSTREAM_COMMIT}"
RAW_FILES = {
    "FinTruthQA.csv": ("dataset/FinTruthQA.csv", "ebde860c012483780897ef0ab2cb2c9437e90a72d774207e460d183a59fef62f"),
    "LICENSE": ("LICENSE", "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"),
}

LABEL_COLS = ["is_question", "ques_relevance", "ans_readability", "ans_relevance"]

# 标签含义（来自官方标注指南《上海和深圳交易所投资者交流平台Q&A 标注指南》第 4 节）
LABEL_MEANING = {
    "is_question": {1: "是真问题", 0: "不是问题（评论/建议/抱怨）"},
    "ques_relevance": {1: "与投资相关", 0: "与投资无关"},
    "ans_readability": {1: "1 读不懂", 2: "2 较难读懂", 3: "3 基本读懂", 4: "4 清楚无歧义"},
    "ans_relevance": {1: "1 完全不相关", 2: "2 部分相关/难判断", 3: "3 完全相关"},
}
LABEL_TITLE = {
    "is_question": "① 提问是不是真问题",
    "ques_relevance": "② 提问是否与投资相关",
    "ans_readability": "③ 回答可读性（1–4）",
    "ans_relevance": "④ 回答相关性（1–3）",
}

# 配色（dataviz 参考调色板，浅色底）
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
ORDINAL_3 = ["#86b6ef", "#2a78d6", "#184f95"]  # 蓝色序数阶 250 / 450 / 600


# ---------------------------------------------------------------- 1. 下载

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_raw() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, (remote, expected) in RAW_FILES.items():
        local = RAW_DIR / name
        if not local.exists():
            url = f"{UPSTREAM_RAW}/{remote}"
            print(f"下载 {url}")
            with urllib.request.urlopen(url, timeout=60) as resp, open(local, "wb") as f:
                shutil.copyfileobj(resp, f)
        actual = sha256(local)
        if actual != expected:
            sys.exit(f"[错误] {local} 的 sha256 对不上：{actual}（期望 {expected}），删掉它重跑。")
    return RAW_DIR / "FinTruthQA.csv"


# ---------------------------------------------------------------- 2. 清洗

# 全角字母、数字、空格和常见数学符号 → 半角；中文标点（，。？！：；“”（）【】）保留不动
_FW_TO_HW = {c: c - 0xFEE0 for r in [(0xFF10, 0xFF19), (0xFF21, 0xFF3A), (0xFF41, 0xFF5A)] for c in range(r[0], r[1] + 1)}
_FW_TO_HW.update({ord(c): ord(c) - 0xFEE0 for c in "％＋－／＝＜＞．"})
_FW_TO_HW.update({0x3000: 0x20, 0x00A0: 0x20})

_CJK = r"[一-鿿　-〿＀-￯“”‘’]"
# 只删真正的 HTML 标签；<COK>、<不存在未披露信息> 这类拿尖括号当书名号的保留
_RE_TAG = re.compile(r"</?(?:a|br|p|span|div|img|font|b|i|strong|em)[^<>]*>", re.IGNORECASE)
_RE_WS = re.compile(r"\s+")
_RE_CJK_SPACE = re.compile(rf"(?<={_CJK}) (?={_CJK})")


def step_html(s: str) -> str:
    return _RE_TAG.sub("", html.unescape(s))


def step_fullwidth(s: str) -> str:
    return s.translate(_FW_TO_HW)


def step_whitespace(s: str) -> str:
    s = _RE_WS.sub(" ", s).strip()
    return _RE_CJK_SPACE.sub("", s)  # 中文字/中文标点之间的空格是噪音，去掉；英文单词、数字之间的保留


CLEAN_STEPS = [
    ("HTML 实体/标签还原", step_html),
    ("全角字母数字空格转半角", step_fullwidth),
    ("空白规范化（换行/制表/连续空格/中文间空格）", step_whitespace),
]

# 纯客套话：去掉这些礼貌用语和标点后几乎什么都不剩的回答
_RE_POLITE = re.compile(
    r"尊敬的|投资者|朋友|先生|女士|您好|你好|感谢|谢谢|多谢|非常|您的|您对|对本?公司|"
    r"关注|支持|建议|关心|理解|厚爱|与|和|及|的|再次|已收悉|收到|回答|答|"
    r"[，。！？、：；,.!?:;~～\s]"
)


def is_template_answer(ans: str) -> bool:
    return _RE_POLITE.sub("", ans) == ""


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log = {"原始行数": len(raw)}
    df = raw.rename(columns={"Unnamed: 0": "id", "QUES": "question", "ANS": "answer"})
    df = df.rename(columns={c: c.lower() for c in df.columns})

    # 2.1 文本规范化，逐步记录每一步改动了多少条
    step_log = {}
    for col in ["question", "answer"]:
        df[col] = df[col].astype("string")
        for name, fn in CLEAN_STEPS:
            before = df[col].copy()
            df[col] = df[col].map(lambda s: fn(s) if isinstance(s, str) else s)
            step_log.setdefault(name, {})[col] = int((before != df[col]).fillna(False).sum())
    log["文本规范化改动条数"] = step_log

    # 2.2 缺失值：NaN 或清洗后变成空串都算缺失
    for col in ["question", "answer"]:
        df.loc[(df[col].str.len() == 0).fillna(False), col] = pd.NA
    missing = {c: int(df[c].isna().sum()) for c in ["question", "answer"] + LABEL_COLS}
    log["缺失值"] = missing
    df = df.dropna(subset=["question", "answer"] + LABEL_COLS)

    # 2.3 标签转数字并校验取值范围
    for col in ["is_question", "ques_relevance"]:
        mapped = df[col].str.strip().str.lower().map({"positive": 1, "negative": 0})
        if mapped.isna().any():
            sys.exit(f"[错误] {col} 出现了 Positive/Negative 以外的值：{df.loc[mapped.isna(), col].unique()}")
        df[col] = mapped.astype(int)
    for col, valid in [("ans_readability", {1, 2, 3, 4}), ("ans_relevance", {1, 2, 3})]:
        df[col] = df[col].astype(int)
        bad = set(df[col].unique()) - valid
        if bad:
            sys.exit(f"[错误] {col} 出现非法取值 {bad}")

    # 2.4 去重：问题+回答完全一样才算重复
    dup_mask = df.duplicated(["question", "answer"], keep="first")
    dup_conflict = df[df.duplicated(["question", "answer"], keep=False)].groupby(["question", "answer"])[LABEL_COLS].nunique().gt(1).any(axis=1).sum()
    log["完全重复(问题+回答)删除"] = int(dup_mask.sum())
    log["重复组中标签互相矛盾"] = int(dup_conflict)
    df = df[~dup_mask].copy()

    # 2.5 长度与标记（只标记、不删除：这是公开 benchmark，删行会让结果没法和论文对比）
    df["q_len"] = df["question"].str.len().astype(int)
    df["a_len"] = df["answer"].str.len().astype(int)
    q_fence = tukey_upper(df["q_len"])
    a_fence = tukey_upper(df["a_len"])
    df["flag_q_long"] = df["q_len"] > q_fence
    df["flag_a_long"] = df["a_len"] > a_fence
    df["flag_a_short"] = df["a_len"] < 5
    df["flag_over_512"] = (df["q_len"] + df["a_len"]) > 509  # 中文 BERT 一字≈一 token，512 减去 3 个特殊 token
    df["flag_dup_question"] = df.duplicated("question", keep=False)  # 同一句话问了不同公司，回答不同，保留
    df["flag_template_answer"] = df["answer"].map(is_template_answer)
    # 标注规则自洽检查：指南规定「回答完全不可读(1) → 相关性也给 1」
    df["flag_label_rule_conflict"] = (df["ans_readability"] == 1) & (df["ans_relevance"] != 1)
    log["长度异常阈值"] = {"问题(Tukey 上界)": q_fence, "回答(Tukey 上界)": a_fence, "回答过短": "< 5 字"}
    log["清洗后行数"] = len(df)

    cols = ["id", "question", "answer"] + LABEL_COLS + ["q_len", "a_len"] + [c for c in df.columns if c.startswith("flag_")]
    return df[cols].reset_index(drop=True), log


def tukey_upper(s: pd.Series) -> float:
    q1, q3 = s.quantile([0.25, 0.75])
    return float(q3 + 1.5 * (q3 - q1))


# ---------------------------------------------------------------- 3. 描述性统计

def describe(df: pd.DataFrame) -> dict:
    n = len(df)
    stats = {"样本数": n}

    stats["标签分布"] = {
        col: {LABEL_MEANING[col][k]: {"条数": int(v), "占比": round(v / n, 4)} for k, v in df[col].value_counts().sort_index().items()}
        for col in LABEL_COLS
    }

    def length_row(s: pd.Series) -> dict:
        return {
            "均值": round(float(s.mean()), 1), "标准差": round(float(s.std()), 1), "最小": int(s.min()),
            "P25": float(s.quantile(0.25)), "中位数": float(s.median()), "P75": float(s.quantile(0.75)),
            "P95": round(float(s.quantile(0.95)), 1), "最大": int(s.max()),
        }

    stats["文本长度(字)"] = {"问题": length_row(df["q_len"]), "回答": length_row(df["a_len"])}
    stats["回答长度按相关性"] = {
        LABEL_MEANING["ans_relevance"][k]: {"条数": int(len(g)), "中位数": float(g["a_len"].median()), "均值": round(float(g["a_len"].mean()), 1)}
        for k, g in df.groupby("ans_relevance")
    }
    # 指南规定：提问是建议/评论时，回答只要表示「收到」就算相关 → 只看真问题时相关性分布才反映「有没有正面回答」
    stats["回答相关性按是否真问题"] = {
        ("真问题" if k == 1 else "不是问题"): {LABEL_MEANING["ans_relevance"][r]: {"条数": int(v), "占比": round(v / len(g), 4)} for r, v in g["ans_relevance"].value_counts().sort_index().items()}
        for k, g in df.groupby("is_question")
    }
    stats["可读性x相关性交叉表"] =pd.crosstab(df["ans_readability"], df["ans_relevance"]).to_dict()
    stats["是否问题x是否相关交叉表"] = pd.crosstab(df["is_question"], df["ques_relevance"]).to_dict()

    tpl = df[df["flag_template_answer"]]
    stats["纯客套话回答"] = {
        "条数": int(len(tpl)), "占比": round(len(tpl) / n, 4),
        "其中相关性=1": int((tpl["ans_relevance"] == 1).sum()),
        "其中相关性=3": int((tpl["ans_relevance"] == 3).sum()),
        "相关性=3 的里面提问不是真问题(建议/评论)": int(((tpl["ans_relevance"] == 3) & (tpl["is_question"] == 0)).sum()),
    }
    stats["标记计数"] = {c: int(df[c].sum()) for c in df.columns if c.startswith("flag_")}

    # 数据集没有公司、日期字段。只能退而求其次：看文本里提到的年份（粗略代理，不等于发帖年份）
    yr = df["answer"].str.extract(r"(20[12]\d)年", expand=False)
    stats["回答中提到的年份(代理)"] = {"有年份的条数": int(yr.notna().sum()), "分布": {k: int(v) for k, v in yr.value_counts().sort_index().items()}}
    return stats


# ---------------------------------------------------------------- 4. 出图

def setup_style():
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": INK2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
        "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    })


def fig_labels(df: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6))
    n = len(df)
    for ax, col in zip(axes.flat, LABEL_COLS):
        vc = df[col].value_counts().sort_index()
        names = [LABEL_MEANING[col][k] for k in vc.index]
        y = np.arange(len(vc))
        ax.barh(y, vc.values, height=0.6, color=BLUE, edgecolor=SURFACE, linewidth=2)
        for yi, v in zip(y, vc.values):
            ax.text(v + n * 0.012, yi, f"{v:,}（{v / n:.1%}）", va="center", fontsize=9.5, color=INK2)
        ax.set_yticks(y, names)
        ax.set_xlim(0, n * 1.08)
        ax.set_title(LABEL_TITLE[col], pad=8)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="both", length=0)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.suptitle(f"FinTruthQA 四个标注维度的标签分布（N = {n:,}）", x=0.01, ha="left", fontsize=13.5, fontweight="bold")
    fig.text(0.01, 0.005, "数据：bethxx99/FinTruthQA（沪深交易所互动平台问答）。标签定义见官方标注指南。", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_lengths(df: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, label, color, bw in [(axes[0], "q_len", "问题", BLUE, 5), (axes[1], "a_len", "回答", ORANGE, 10)]:
        s = df[col]
        cap = float(np.ceil(s.quantile(0.99) / bw) * bw)
        shown = s[s <= cap]
        ax.hist(shown, bins=np.arange(0, cap + bw, bw), color=color, edgecolor=SURFACE, linewidth=1)
        med, mean = s.median(), s.mean()
        top = ax.get_ylim()[1] * 1.15  # 顶部留白放标注，免得压在柱子上
        ax.set_ylim(0, top)
        # 中位数标在线左边、均值标在线右边，两条线挨得近也不会压字
        for x, txt, ls, side in [(med, f"中位数 {med:.0f}", "-", -1), (mean, f"均值 {mean:.0f}", "--", 1)]:
            ax.axvline(x, color=INK, linewidth=1, linestyle=ls)
            ax.text(x + side * cap * 0.012, top * 0.97, txt, fontsize=9, color=INK, va="top", ha="left" if side > 0 else "right")
        ax.set_title(f"{label}长度（字）", pad=8)
        ax.set_xlabel("字数")
        ax.set_ylabel("条数")
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        hidden = int((s > cap).sum())
        ax.text(0.99, 0.6, f"最长 {s.max():,} 字\n超过 {cap:.0f} 字的 {hidden} 条（{hidden / len(s):.1%}）未画出", transform=ax.transAxes, ha="right", fontsize=8.5, color=MUTED)
    fig.suptitle("问题短、回答长：回答长度的长尾明显更重", x=0.01, ha="left", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_answer_quality(df: pd.DataFrame, path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})

    # 左：回答长度按相关性分组
    levels = [1, 2, 3]
    data = [df.loc[df["ans_relevance"] == k, "a_len"] for k in levels]
    bp = ax1.boxplot(data, vert=True, widths=0.5, showfliers=False, patch_artist=True,
                     medianprops={"color": INK, "linewidth": 1.5}, whiskerprops={"color": AXIS}, capprops={"color": AXIS})
    for patch, c in zip(bp["boxes"], ORDINAL_3):
        patch.set(facecolor=c, edgecolor=SURFACE, linewidth=2)
    for i, d in enumerate(data, start=1):
        ax1.text(i + 0.3, d.median(), f"中位数 {d.median():.0f}", va="center", fontsize=9, color=INK)
    ax1.set_xticks(levels, [f"{LABEL_MEANING['ans_relevance'][k]}\nn={len(d):,}" for k, d in zip(levels, data)])
    ax1.set_ylabel("回答字数")
    ax1.set_title("敷衍的回答最短，部分相关的反而最长", pad=8)
    ax1.yaxis.grid(True)
    ax1.set_axisbelow(True)
    ax1.tick_params(length=0)
    ax1.text(0.0, -0.22, "箱体=P25–P75，须=1.5 倍四分位距，未画离群点", transform=ax1.transAxes, fontsize=8.5, color=MUTED)

    # 右：可读性 × 相关性交叉热力图
    ct = pd.crosstab(df["ans_readability"], df["ans_relevance"]).reindex(index=[4, 3, 2, 1], columns=levels, fill_value=0)
    ax2.imshow(np.log1p(ct.values), cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#eef4fd", "#104281"]), aspect="auto")
    vmax = np.log1p(ct.values).max()
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = ct.values[i, j]
            ax2.text(j, i, f"{v:,}", ha="center", va="center", fontsize=10, color="#ffffff" if np.log1p(v) > vmax * 0.6 else INK)
    ax2.set_xticks(range(3), ["1 不相关", "2 部分相关", "3 完全相关"])
    ax2.set_yticks(range(4), ["4 清楚", "3 基本读懂", "2 较难读懂", "1 读不懂"])
    ax2.set_xlabel("回答相关性")
    ax2.set_ylabel("回答可读性")
    ax2.set_title("可读性与相关性高度绑定（条数）", pad=8)
    for s in ax2.spines.values():
        s.set_visible(False)
    ax2.tick_params(length=0)

    fig.suptitle("回答质量两个维度：和长度的关系、彼此的关系", x=0.01, ha="left", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- 5. 报告

def md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    def fmt(v):
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        if isinstance(v, (float, np.floating)):
            return f"{int(v):,}" if float(v).is_integer() else f"{v:,.1f}"
        return str(v)

    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def write_report(df: pd.DataFrame, log: dict, stats: dict, path: Path):
    n = len(df)
    parts = ["# FinTruthQA 清洗与描述统计报告（脚本自动生成，勿手改）", ""]
    parts += ["## 清洗日志", "", f"- 原始行数：{log['原始行数']:,}；清洗后：{log['清洗后行数']:,}",
              f"- 缺失值：{log['缺失值']}",
              f"- 完全重复（问题+回答）删除：{log['完全重复(问题+回答)删除']}；重复组中标签矛盾：{log['重复组中标签互相矛盾']}",
              f"- 长度异常阈值：{log['长度异常阈值']}", ""]
    steps = pd.DataFrame([{"步骤": k, "问题改动条数": v["question"], "回答改动条数": v["answer"]} for k, v in log["文本规范化改动条数"].items()])
    parts += [md_table(steps), ""]

    parts += ["## 标签分布", ""]
    for col in LABEL_COLS:
        t = pd.DataFrame([{"标签": k, "条数": f"{v['条数']:,}", "占比": f"{v['占比']:.1%}"} for k, v in stats["标签分布"][col].items()])
        parts += [f"### {LABEL_TITLE[col]}（`{col}`）", "", md_table(t), ""]

    parts += ["## 文本长度（字）", ""]
    t = pd.DataFrame(stats["文本长度(字)"]).T.reset_index().rename(columns={"index": "字段"})
    parts += [md_table(t), ""]
    t = pd.DataFrame(stats["回答长度按相关性"]).T.reset_index().rename(columns={"index": "回答相关性"})
    parts += ["### 回答长度按相关性分组", "", md_table(t), ""]

    t = pd.DataFrame([{"提问类型": q, "回答相关性": r, "条数": v["条数"], "占比": f"{v['占比']:.1%}"}
                      for q, d in stats["回答相关性按是否真问题"].items() for r, v in d.items()])
    parts += ["## 回答相关性按「提问是不是真问题」分组", "", md_table(t), ""]

    ct = pd.crosstab(df["ans_readability"], df["ans_relevance"])
    ct.index =[f"可读性 {i}" for i in ct.index]
    ct.columns = [f"相关性 {c}" for c in ct.columns]
    parts += ["## 可读性 × 相关性", "", md_table(ct.reset_index().rename(columns={"index": ""})), ""]
    ct = pd.crosstab(df["is_question"].map({1: "是问题", 0: "不是问题"}), df["ques_relevance"].map({1: "与投资相关", 0: "无关"}))
    parts += ["## 是否问题 × 是否与投资相关", "", md_table(ct.reset_index().rename(columns={"is_question": ""})), ""]

    parts += ["## 纯客套话回答", "", f"`{json.dumps(stats['纯客套话回答'], ensure_ascii=False)}`", ""]
    flags = pd.DataFrame([{"标记": k, "条数": v, "占比": f"{v / n:.1%}"} for k, v in stats["标记计数"].items()])
    parts += ["## 标记列计数（只标记，不删除）", "", md_table(flags), ""]
    yr = stats["回答中提到的年份(代理)"]
    parts += ["## 回答中提到的年份（代理，非发帖时间）", "", f"有年份的条数：{yr['有年份的条数']:,}", "",
              md_table(pd.DataFrame([{"年份": k, "条数": v} for k, v in yr["分布"].items()])), ""]
    path.write_text("\n".join(parts), encoding="utf-8")


# ---------------------------------------------------------------- main

def main():
    for d in (OUT_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    raw_path = fetch_raw()
    raw = pd.read_csv(raw_path, encoding="utf-8")
    print(f"读入原始数据 {raw.shape[0]:,} 行 × {raw.shape[1]} 列：{list(raw.columns)}")

    df, log = clean(raw)
    stats = describe(df)

    df.to_csv(OUT_DIR / "fintruthqa_clean.csv", index=False, encoding="utf-8-sig")
    shutil.copyfile(RAW_DIR / "LICENSE", OUT_DIR / "LICENSE-FinTruthQA.txt")  # 再分发数据要随附上游 Apache-2.0 许可
    (OUT_DIR / "summary.json").write_text(json.dumps({"清洗日志": log, "统计": stats}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(df, log, stats, OUT_DIR / "stats_report.md")

    setup_style()
    fig_labels(df, FIG_DIR / "fig1_label_distribution.png")
    fig_lengths(df, FIG_DIR / "fig2_text_length.png")
    fig_answer_quality(df, FIG_DIR / "fig3_answer_quality.png")

    print(f"清洗后 {len(df):,} 行；产出在 {OUT_DIR} 和 {FIG_DIR}")
    print(json.dumps(log, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
