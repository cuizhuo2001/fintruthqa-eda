"""基线 1：TF-IDF + 逻辑回归，预测回答相关性（1/2/3）。

特征 = 问题的字 n-gram TF-IDF + 回答的字 n-gram TF-IDF + 「是否真问题」标签（和 LLM 拿到的信息一样）。
类别不平衡用 class_weight="balanced"；正则强度 C 在训练集上 5 折交叉验证按 macro-F1 选。

python -I eval_tfidf.py
"""

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I 不会自动把脚本目录加进来
from eval_common import EVAL_DIR, SEED, TARGET, get_split, metrics, save_predictions, write_json  # noqa: E402


class Passthrough(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return sp.csr_matrix(np.asarray(X, dtype=float))


def build_model():
    def tfidf():
        # 中文不分词，直接用 1–3 个字的组合当特征
        return TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=2, sublinear_tf=True, max_features=50000)

    features = ColumnTransformer([
        ("q", tfidf(), "question"),
        ("a", tfidf(), "answer"),
        ("is_q", Passthrough(), ["is_question"]),
    ])
    clf = LogisticRegression(class_weight="balanced", max_iter=3000)
    return Pipeline([("features", features), ("clf", clf)])


def main():
    train, test = get_split()
    print(f"训练集 {len(train)} 条，测试集 {len(test)} 条")

    search = GridSearchCV(
        build_model(),
        {"clf__C": [0.3, 1, 3, 10, 30]},
        scoring="f1_macro",
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        n_jobs=2,  # 本机 8G 内存，别开太多进程
    )
    search.fit(train, train[TARGET])
    print(f"交叉验证选出 C={search.best_params_['clf__C']}，CV macro-F1={search.best_score_:.3f}")

    pred = search.predict(test)
    save_predictions("tfidf_lr", test["id"], pred)
    m = metrics(test[TARGET], pred)
    print(f"测试集 macro-F1={m['macro_f1']:.3f}  Kappa={m['kappa']:.3f}  准确率={m['accuracy']:.3f}")
    write_json(EVAL_DIR / "tfidf_lr_meta.json", {
        "best_C": search.best_params_["clf__C"],
        "cv_macro_f1": search.best_score_,
        "cv_results": {str(c): float(s) for c, s in zip(search.cv_results_["param_clf__C"], search.cv_results_["mean_test_score"])},
    })


if __name__ == "__main__":
    main()
