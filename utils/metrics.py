import os
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score


def compute_metrics_emotion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """
    计算情绪识别指标：
    - UAR (Unweighted Average Recall)
    - WAR (Accuracy)
    - Macro-F1
    """
    cm = confusion_matrix(y_true, y_pred)
    war = accuracy_score(y_true, y_pred)

    # 每类召回率 = TP / (TP + FN)
    row_sums = np.sum(cm, axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        recall_per_class = np.diag(cm) / row_sums
    recall_per_class = np.nan_to_num(recall_per_class)  # 处理除零情况

    uar = np.mean(recall_per_class)
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    return uar, war, macro_f1


from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def compute_metrics_cognitive(y_true, y_pred):
    """
    针对认知障碍筛查二分类的评估函数
    类别 0: 健康 (HC)
    类别 1: 轻度认知障碍 (MCI)
    """
    # 1. 核心机器学习指标
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    # 2. 临床医学常用指标
    # tn: True Negative (正确预测的健康人)
    # fp: False Positive (误诊为MCI的健康人)
    # fn: False Negative (漏诊的MCI患者)
    # tp: True Positive (正确预测的MCI患者)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # 也就是类别 1 的 Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # 也就是类别 0 的 Recall

    return acc, macro_f1


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], classes: List[str], save_path: str) -> None:
    """
    绘制并保存混淆矩阵
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # 坐标轴设置
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)

    # 填充数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


class EarlyStopping:
    """
    如果验证集指标在设定的 patience 个 Epoch 内没有改进，则触发早停。
    我们通常监控 UAR + WAR 的和，因此 mode='max'。
    """

    def __init__(self, patience: int = 7, min_delta: float = 0, mode: str = 'max', verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

        # 确定比较函数：'max' -> score > best + delta; 'min' -> score < best - delta
        if self.mode == 'min':
            self.best_score = np.Inf
            self.is_better = lambda score, best: score < best - self.min_delta
        else:  # mode == 'max'
            self.best_score = -np.Inf
            self.is_better = lambda score, best: score > best + self.min_delta

    def __call__(self, current_metric: float) -> bool:
        """
        输入当前验证指标。返回 True 表示应该停止训练。
        """
        score = current_metric

        if self.best_score is None:
            self.best_score = score

        elif self.is_better(score, self.best_score):
            # 性能提升，重置计数器，更新最佳分数
            self.best_score = score
            self.counter = 0
        else:
            # 性能没有提升
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience} (Best Score: {self.best_score:.4f})")

            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


def compute_metrics_regression(y_true, y_pred):
    """计算回归任务的 CCC, RMSE 和 MAE 指标"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # MAE
    mae = np.mean(np.abs(y_true - y_pred))

    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # CCC (Concordance Correlation Coefficient)
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)

    if sd_pred == 0 or sd_true == 0:
        cor = 0.0
    else:
        cor = np.corrcoef(y_true, y_pred)[0, 1]

    # 如果 corrcoef 仍然算出 nan (极罕见情况)，强制置 0
    if np.isnan(cor):
        cor = 0.0

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    ccc = numerator / denominator if denominator != 0 else 0.0

    return ccc, rmse, mae
