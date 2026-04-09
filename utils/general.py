import os
import sys
import random
import logging
import json

import torch
import numpy as np


def set_seed(seed: int = 42) -> None:
    """设定全局随机种子，保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(log_dir: str, name: str = "TrainExperiment") -> logging.Logger:
    """获取 logger，同时输出到控制台和文件"""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # 文件输出
        fh = logging.FileHandler(os.path.join(log_dir, "train.log"), mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # 控制台输出
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def ensure_dir(path: str):
    """确保目录存在，不存在则创建"""
    if not os.path.exists(path):
        os.makedirs(path)


from typing import Dict, Any


def save_checkpoint(state: Dict[str, Any], is_best: bool, checkpoint_dir: str,
                    filename: str = 'checkpoint.pth') -> None:
    """
    保存训练状态的检查点。

    Args:
        state: 包含所有状态的字典 (e.g., epoch, model_state_dict, optimizer_state_dict)。
        is_best: 当前模型是否是最佳模型。
        checkpoint_dir: 检查点保存的目录。
        filename: 通用的检查点文件名。
    """
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    # 如果是最佳模型，额外保存一份名为 model_best.pth
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'model_best.pth')
        torch.save(state, best_filepath)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """加载检查点恢复训练"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    start_epoch = checkpoint.get('epoch', 0) + 1
    best_metric = checkpoint.get('best_metric', 0.0)

    return start_epoch, best_metric


class NumpyEncoder(json.JSONEncoder):
    """自定义 JSON 编码器，用于处理 numpy 数据类型"""

    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)