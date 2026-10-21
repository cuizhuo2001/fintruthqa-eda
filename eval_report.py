"""汇总：把各模型在测试集上的预测和人工标注比，出对比表和图。

python -I eval_report.py

读 output/eval/predictions/*.csv（哪个模型的预测文件在就比哪个），产出：
  output/eval/results.md     对比表（README 里的数都来自这里）
  output/eval/results.json   同样的数字，机器可读
  figures/fig4_baseline_vs_llm.png   macro-F1 和 Kappa 对比（带 95% 置信区间）
  figures/fig5_confusion.png         各模型的混淆矩阵
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I 不会自动把脚本目录加进来
from eval_common import EVAL_DIR, FIG_DIR, LABELS, PRED_DIR, TARGET, bootstrap_ci, get_split, metrics, write_json  # noqa: E402
from fintruthqa_pipeline import AXIS, BLUE, INK2, MUTED, ORANGE, SURFACE, setup_style  # noqa: E402

# 预测文件名 → (表里显示的名字, 类别)
MODELS = {
    "majority": ("多数类（全猜 3 分）", "参照"),
    "tfidf_lr": ("TF-IDF + 逻辑回归", "基线"),
    "bert": ("中文 BERT 微调", "基线"),
    "llm_zero_shot": ("DeepSeek 零样本", "LLM"),
    "llm_few_shot": ("DeepSeek 少样本（8 例）", "LLM"),
}
FAMILY_COLOR = {"参照": AXIS, "基线": BLUE, "LLM": ORANGE}


def collect(test):
    y = test.set_index("id")[TARGET]
    preds = {"majority": pd.Series(3, index=y.index)}
    for name in MODELS:
        path = PRED_DIR / f"{name}.csv"
        if path.exists():
            p = pd.read_csv(path).set_index("id")["pred"]
            preds[name] = p.reindex(y.index)
    return y, {k: preds[k] for k in MODELS if k in preds}


def results_table(y, preds, test):
    rows = []
    is_q = test.set_index("id")["is_question"].reindex(y.index) == 1
    for name, p in preds.items():
        m = metrics(y.values, p.values)
        m.update(bootstrap_ci(y.values, p.values))
        mq = metrics(y[is_q].values, p[is_q].values)
        m["macro_f1_real_questions"] = mq["macro_f1"]
        m["kappa_real_questions"] = mq["kappa"]
        rows.append({"model": name, "label": MODELS[name][0], "family": MODELS[name][1], **m})
    return pd.DataFrame(rows)


def write_markdown(res, test, path):
    def ci(lo_hi):
        return f"[{lo_hi[0]:.3f}, {lo_hi[1]:.3f}]"

    dist = test[TARGET].value_counts().sort_index()
    lines = [
        "# 基线 vs LLM：回答相关性（1/2/3）打分结果",
        "",
        "由 `eval_report.py` 自动生成。",
        "",
        f"测试集 {len(test):,} 条（从 6,000 条里按「相关性 × 是否真问题」分层抽 20%，随机种子 42，"
        f"划分存在 `output/eval/split.csv`）。人工标注分布：1 分 {dist[1]} 条、2 分 {dist[2]} 条、3 分 {dist[3]} 条。",
        "",
        "## 总表",
        "",
        "| 模型 | macro-F1 | 95% 置信区间 | Cohen's Kappa | 95% 置信区间 | 准确率 | 加权 Kappa（二次） |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in res.iterrows():
        lines.append(f"| {r['label']} | {r['macro_f1']:.3f} | {ci(r['macro_f1_ci'])} | {r['kappa']:.3f} | "
                     f"{ci(r['kappa_ci'])} | {r['accuracy']:.3f} | {r['kappa_quadratic']:.3f} |")
    lines += [
        "",
        "## 分类别 F1",
        "",
        "| 模型 | 1 分（完全没回答） | 2 分（部分回答） | 3 分（全答了） |",
        "|---|---|---|---|",
    ]
    for _, r in res.iterrows():
        lines.append(f"| {r['label']} | {r['f1_class1']:.3f} | {r['f1_class2']:.3f} | {r['f1_class3']:.3f} |")
    n_real = int((test["is_question"] == 1).sum())
    lines += [
        "",
        f"## 只看真问题（{n_real:,} 条）",
        "",
        "| 模型 | macro-F1 | Cohen's Kappa |",
        "|---|---|---|",
    ]
    for _, r in res.iterrows():
        lines.append(f"| {r['label']} | {r['macro_f1_real_questions']:.3f} | {r['kappa_real_questions']:.3f} |")
    lines += [
        "",
        "说明：Kappa 是模型和人工标注的一致性，扣掉了「瞎猜也能对上」的部分，0 = 跟瞎猜一样，1 = 完全一致。"
        "置信区间是在测试集上有放回重抽 1,000 次算出来的。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def fig_metrics(res, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    y = np.arange(len(res))[::-1]
    for ax, col, title in [(axes[0], "macro_f1", "macro-F1（三类 F1 的平均）"),
                           (axes[1], "kappa", "Cohen's Kappa（与人工标注的一致性）")]:
        colors = [FAMILY_COLOR[f] for f in res["family"]]
        ax.barh(y, res[col], height=0.6, color=colors, edgecolor=SURFACE, linewidth=2)
        lo = res[col] - res[f"{col}_ci"].str[0]
        hi = res[f"{col}_ci"].str[1] - res[col]
        ax.errorbar(res[col], y, xerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
        for yi, v, h in zip(y, res[col], res[f"{col}_ci"].str[1]):
            ax.text(h + 0.015, yi, f"{v:.3f}", va="center", fontsize=9.5, color=INK2)
        ax.set_xlim(0, 1)
        ax.set_title(title, pad=8)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="both", length=0)
    axes[0].set_yticks(y, res["label"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLOR.values()]
    fig.legend(handles, FAMILY_COLOR.keys(), loc="upper right", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("回答相关性打分：基线 vs LLM（测试集 1,200 条）", x=0.01, ha="left", fontsize=13.5, fontweight="bold")
    fig.text(0.01, 0.01, "误差线 = 95% bootstrap 置信区间。多数类 = 全部猜 3 分，Kappa 恒为 0，作为参照。",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_confusion(y, preds, path):
    names = [n for n in preds if n != "majority"]
    cmap = LinearSegmentedColormap.from_list("blue", [SURFACE, BLUE, "#184f95"])
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.9))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        cm = confusion_matrix(y.values, preds[name].values, labels=LABELS)
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{share[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center", fontsize=8.5,
                        color="white" if share[i, j] > 0.4 else INK2)
        ax.set_xticks(range(3), LABELS)
        ax.set_yticks(range(3), LABELS)
        ax.set_xlabel("模型打分")
        ax.set_title(MODELS[name][0], fontsize=10)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("人工标注")
    fig.suptitle("混淆矩阵（每行按人工标注归一化）", x=0.01, ha="left", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main():
    _, test = get_split()
    y, preds = collect(test)
    res = results_table(y, preds, test)
    write_markdown(res, test, EVAL_DIR / "results.md")
    write_json(EVAL_DIR / "results.json", res.to_dict(orient="records"))
    setup_style()
    fig_metrics(res, FIG_DIR / "fig4_baseline_vs_llm.png")
    fig_confusion(y, preds, FIG_DIR / "fig5_confusion.png")
    print((EVAL_DIR / "results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
