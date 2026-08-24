# -*- coding: utf-8 -*-
"""
第7章 文本分类 —— 基于 SVM 与 Logistic Regression 的中文垃圾邮件分类

【说明】
- 本实验不使用朴素贝叶斯（MultinomialNB）和 TextCNN 模型。
- 采用 SVM（线性支持向量机）作为主分类器，并与逻辑回归（Logistic Regression）对比。
- 使用 jieba 中文分词 + TF-IDF 特征提取 + sklearn Pipeline 构建完整分类流程。

数据集：实验四的中文邮件数据集（0~150.txt 为已标注数据，151~155.txt 为待预测数据）
  标注规则：0~126.txt → 垃圾邮件(spam=1)；127~150.txt → 普通邮件(ham=0)
"""

import re
from pathlib import Path

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer    # TF-IDF 特征提取
from sklearn.model_selection import train_test_split          # 数据集划分
from sklearn.svm import LinearSVC                             # 线性 SVM 分类器
from sklearn.linear_model import LogisticRegression           # 逻辑回归分类器
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.pipeline import Pipeline                         # 管道：串联分词→特征→模型

# ============================================================
# 1. 配置路径与参数
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
# 从 Homework/CH07/ 向上两层到 NLP 根目录，再进入 lab/lab04/bayes-mails-classify-master/邮件_files/
EMAIL_DIR = BASE_DIR.parent.parent / "lab" / "lab04" / "bayes-mails-classify-master" / "邮件_files"

RANDOM_STATE = 42        # 随机种子，保证结果可复现
TEST_SIZE = 0.3          # 测试集占比 30%

# 用于演示预测的自定义文本
DEMO_TEXTS = [
    "免费领取内部资料并加入股票群，名额有限速来",
    "老师您好，明天下午三点在学院会议室开组会，请准时参加",
    "恭喜您获得苹果手机一部，点击链接填写收货地址即可领取",
    "实验报告我已经写好了，麻烦老师帮我看一下有没有问题",
]


# ============================================================
# 2. 文本预处理 —— 中文分词
# ============================================================
def tokenize(text: str) -> list[str]:
    """
    中文文本预处理与分词。
    步骤：
      1) 用正则去掉标点、数字、特殊符号等无效字符；
      2) 使用 jieba 精确模式（lcut）切词；
      3) 过滤掉单字 token（长度为1的词通常语义信息不足）。
    """
    # 去掉标点、数字、特殊符号（保留中文字符和字母）
    text = re.sub(r'[.【】0-9、——。，！~\*\n\r\t\s]', '', text)
    # jieba 精确模式分词
    tokens = jieba.lcut(text)
    # 过滤：仅保留长度 > 1 的 token
    tokens = [t for t in tokens if len(t) > 1]
    return tokens


# ============================================================
# 3. 数据加载 —— 从邮件文件中读取文本
# ============================================================
def load_email_files(email_dir: Path) -> tuple[list[str], list[int]]:
    """
    从邮件文件夹中读取所有已标注的邮件文本。
    加载 0.txt ~ 150.txt（共151封邮件），其中：
      - 0~126.txt  标记为垃圾邮件（spam, label=1）
      - 127~150.txt 标记为普通邮件（ham,  label=0）

    返回值：
      texts:  文本内容列表
      labels: 标签列表（1=spam, 0=ham）
    """
    texts: list[str] = []
    labels: list[int] = []

    for i in range(151):  # 0~150 号文件
        file_path = email_dir / f"{i}.txt"
        if not file_path.exists():
            print(f"警告：文件 {file_path} 不存在，跳过。")
            continue
        with file_path.open("r", encoding="utf-8-sig") as f:
            content = f.read().strip()
        texts.append(content)
        # 前127封（0~126）为垃圾邮件，后24封（127~150）为普通邮件
        labels.append(1 if i <= 126 else 0)

    return texts, labels


def load_test_files(email_dir: Path) -> list[str]:
    """
    加载待预测的测试邮件（151~155号，共5封）。
    """
    test_texts: list[str] = []
    for i in range(151, 156):
        file_path = email_dir / f"{i}.txt"
        if file_path.exists():
            with file_path.open("r", encoding="utf-8-sig") as f:
                test_texts.append(f.read().strip())
    return test_texts


# ============================================================
# 4. 构建分类 Pipeline
# ============================================================
def build_pipeline(classifier) -> Pipeline:
    """
    构建 sklearn Pipeline，将以下步骤串联为一个整体：

    Step 1 — TfidfVectorizer（TF-IDF 特征提取）：
        - 使用自定义 jieba 分词器（tokenizer=tokenize）
        - token_pattern=None：因为使用自定义分词器，必须关闭内置正则
        - max_df=0.9：忽略出现在90%以上文档中的词（近似停用词过滤）
        - min_df=2：忽略仅在1个文档中出现的低频词（减少噪声）
        - ngram_range=(1,2)：同时使用一元词组和二元词组，捕获局部短语信息
          例如："免费 领取" 这样的二元组合比单独的"免费"和"领取"更具判别力

    Step 2 — clf（分类器）：
        - SVM (LinearSVC)：在线性可分假设下寻找最大间隔超平面，适合高维稀疏文本特征
        - Logistic Regression：通过 sigmoid 函数输出类别概率，参数可解释性强
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            max_df=0.9,
            min_df=2,
            ngram_range=(1, 2),
        )),
        ("clf", classifier),
    ])


# ============================================================
# 5. 模型训练与评估
# ============================================================
def train_and_evaluate(
    model_name: str,
    pipeline: Pipeline,
    x_train: list[str],
    x_test: list[str],
    y_train: list[int],
    y_test: list[int],
) -> dict:
    """
    在训练集上训练模型，在测试集上评估并打印分类指标。
    """
    # ---- 训练 ----
    pipeline.fit(x_train, y_train)

    # ---- 预测 ----
    y_pred = pipeline.predict(x_test)

    # ---- 评估指标 ----
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report_str = classification_report(
        y_test, y_pred,
        target_names=["ham(普通)", "spam(垃圾)"],
        zero_division=0,
    )

    print(f"\n{'='*60}")
    print(f"  {model_name} 分类结果")
    print(f"{'='*60}")
    print(f"准确率 (Accuracy): {acc:.4f}")
    print(f"\n混淆矩阵 (Confusion Matrix):")
    print(f"                预测ham    预测spam")
    print(f"  真实ham        {cm[0][0]:6d}      {cm[0][1]:6d}")
    print(f"  真实spam       {cm[1][0]:6d}      {cm[1][1]:6d}")
    print(f"\n分类报告 (Classification Report):")
    print(report_str)

    return {
        "model": model_name,
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "report": report_str,
    }


# ============================================================
# 6. 对新邮件进行预测演示
# ============================================================
def demo_predict(pipeline: Pipeline, texts: list[str], labels_map: dict[int, str]) -> None:
    """
    使用训练好的模型对未知邮件/自定义文本进行预测并打印分类结果。
    """
    print(f"\n{'='*60}")
    print(f"  演示：对新文本进行预测")
    print(f"{'='*60}")
    for idx, text in enumerate(texts):
        pred = pipeline.predict([text])[0]
        label_name = labels_map[pred]
        # 只显示文本前60个字符避免输出过长
        preview = text.replace('\n', ' ').replace('\r', ' ')[:60]
        print(f"  [{label_name}] {preview}..." if len(text) > 60 else f"  [{label_name}] {preview}")


# ============================================================
# 7. 主流程
# ============================================================
def main() -> None:
    print("=" * 60)
    print("  第7章 文本分类 —— SVM 与 Logistic Regression")
    print("  中文垃圾邮件分类")
    print("=" * 60)

    # ---------- 7.1 加载已标注数据 ----------
    print(f"\n邮件目录: {EMAIL_DIR}")
    texts, labels = load_email_files(EMAIL_DIR)
    print(f"已标注邮件总数: {len(texts)} 封")
    print(f"  垃圾邮件(spam=1): {sum(labels)} 封 (0~126.txt)")
    print(f"  普通邮件(ham=0):  {len(labels) - sum(labels)} 封 (127~150.txt)")

    # 检查类别数
    if len(set(labels)) < 2:
        raise ValueError("数据集中类别数不足 2 类，无法进行二分类。")

    # ---------- 7.2 划分训练集/测试集 ----------
    # stratify=labels 分层采样 → 保证训练/测试集中类别比例一致
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    print(f"\n数据划分 (test_size={TEST_SIZE}):")
    print(f"  训练集: {len(x_train)} 封")
    print(f"  测试集: {len(x_test)} 封")

    # ---------- 7.3 构建并训练 SVM 模型 ----------
    # LinearSVC: 线性支持向量机，适合高维稀疏特征（如文本 TF-IDF）
    # C: 正则化系数，C越小正则化越强（防止过拟合）
    # dual=False: 当样本数 > 特征数时建议使用 primal 形式
    # max_iter: 最大迭代次数，适当增大以确保高维数据收敛
    svm_pipeline = build_pipeline(
        LinearSVC(C=1.0, max_iter=3000, random_state=RANDOM_STATE, dual=False)
    )
    svm_results = train_and_evaluate(
        "SVM (LinearSVC)", svm_pipeline,
        x_train, x_test, y_train, y_test
    )

    # ---------- 7.4 构建并训练 Logistic Regression 模型 ----------
    # LogisticRegression: 对数线性模型，通过 sigmoid 输出概率
    # 与 SVM 同为线性分类器，但优化目标不同（最大化似然 vs 最大化间隔）
    lr_pipeline = build_pipeline(
        LogisticRegression(C=1.0, max_iter=3000, random_state=RANDOM_STATE)
    )
    lr_results = train_and_evaluate(
        "Logistic Regression", lr_pipeline,
        x_train, x_test, y_train, y_test
    )

    # ---------- 7.5 模型对比 ----------
    print(f"\n{'='*60}")
    print(f"  模型对比总结")
    print(f"{'='*60}")
    print(f"  {'模型':<30s} {'准确率':>10s}")
    print(f"  {'-'*42}")
    print(f"  {'SVM (LinearSVC)':<30s} {svm_results['accuracy']:>10.4f}")
    print(f"  {'Logistic Regression':<30s} {lr_results['accuracy']:>10.4f}")

    # ---------- 7.6 对未知邮件（151~155.txt）进行预测 ----------
    unlabeled_texts = load_test_files(EMAIL_DIR)
    if unlabeled_texts:
        print(f"\n加载待预测邮件: {len(unlabeled_texts)} 封 (151~155.txt)")
        label_map = {1: "垃圾邮件(spam)", 0: "普通邮件(ham)"}
        print("\n>>> SVM 模型预测结果:")
        demo_predict(svm_pipeline, unlabeled_texts, label_map)
        print("\n>>> Logistic Regression 模型预测结果:")
        demo_predict(lr_pipeline, unlabeled_texts, label_map)

    # ---------- 7.7 对自定义文本进行演示预测 ----------
    print("\n\n>>> SVM 模型对自定义文本的预测结果:")
    demo_predict(svm_pipeline, DEMO_TEXTS, label_map)

    print(f"\n{'='*60}")
    print("  流程结束。")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
