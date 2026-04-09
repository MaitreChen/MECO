import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataloader.cognition_dataset import MECO_SI_COG_Dataset
from utils.general import get_logger
from utils.metrics import compute_metrics_cognitive, compute_metrics_regression, EarlyStopping
from models.baseline import FusionNet


class TrainerSICog:
    """
    Unified Trainer for Subject-Independent Cognitive Tasks (MMSE).
    Handles both Binary Classification and Regression based on args.task.
    """

    def __init__(self, args, full_data_dict, train_indices, val_indices, save_dir, val_sub):
        self.args = args
        self.task = args.task  # 'cls' or 'reg'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.logger = get_logger(self.save_dir)
        self.writer = SummaryWriter(self.save_dir)
        self.val_sub = val_sub

        # Config properties based on task
        self.num_classes = 2 if self.task == 'cls' else 1

        # 1. Modality Configuration
        self.used_modalities = getattr(args, "used_modalities", ["video", "eeg", "ecg"])
        self.feat_type_video = getattr(args, 'feature_type_video',
                                       'deep_feature') if 'video' in self.used_modalities else None
        self.feat_type_eeg = getattr(args, 'feature_type_eeg',
                                     'eeg_de_feats') if 'eeg' in self.used_modalities else None
        self.feat_type_ecg = getattr(args, 'feature_type_ecg',
                                     'ecg_time_feats') if 'ecg' in self.used_modalities else None

        self._log_modality_configuration()

        # 2. Dataset Instantiation
        self.train_dataset = MECO_SI_COG_Dataset(
            data_source=full_data_dict, task=self.task,
            feat_type_video=self.feat_type_video, feat_type_eeg=self.feat_type_eeg, feat_type_ecg=self.feat_type_ecg,
            sample_indices=train_indices
        )

        # Retrieve normalization statistics computed on the training set
        train_mean_v, train_std_v = self.train_dataset.s_mean_v_dict, self.train_dataset.s_std_v_dict
        train_mean_e, train_std_e = self.train_dataset.s_mean_e_dict, self.train_dataset.s_std_e_dict
        train_mean_c, train_std_c = self.train_dataset.s_mean_c_dict, self.train_dataset.s_std_c_dict

        self.val_dataset = MECO_SI_COG_Dataset(
            data_source=full_data_dict, task=self.task,
            feat_type_video=self.feat_type_video, feat_type_eeg=self.feat_type_eeg, feat_type_ecg=self.feat_type_ecg,
            sample_indices=val_indices,
            s_mean_v=train_mean_v, s_std_v=train_std_v, s_mean_e=train_mean_e, s_std_e=train_std_e,
            s_mean_c=train_mean_c, s_std_c=train_std_c
        )

        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=False)

        # 3. Model Architecture Construction
        active_feature_types = []
        if self.feat_type_video: active_feature_types.append(self.feat_type_video)
        if self.feat_type_eeg: active_feature_types.append(self.feat_type_eeg)
        if self.feat_type_ecg: active_feature_types.append(self.feat_type_ecg)

        self.model = FusionNet(feature_types=active_feature_types, num_classes=self.num_classes, use_gru=False,
                               dropout_rate=0.0).to(self.device)

        # 4. Optimization & Task-Specific Criteria
        if self.task == 'cls':
            self.criterion = nn.CrossEntropyLoss()
            self.best_metric = 0.0  # ACC is primary (Higher is better)
            self.early_stopper = EarlyStopping(patience=args.patience, mode="max")
        else:
            self.criterion = nn.MSELoss()
            self.best_metric = -float('inf')  # CCC is primary (Higher is better)
            self.early_stopper = EarlyStopping(patience=args.patience, mode="max")

        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=1e-6)

    def _log_modality_configuration(self):
        self.logger.info(f"=== SI Fold: {self.val_sub} Configuration ({self.task.upper()}) ===")
        if self.feat_type_video: self.logger.info(f"[*] Video: {self.feat_type_video}")
        if self.feat_type_eeg:   self.logger.info(f"[*] EEG  : {self.feat_type_eeg}")
        if self.feat_type_ecg:   self.logger.info(f"[*] ECG  : {self.feat_type_ecg}")
        self.logger.info("===============================================")

    def train_one_epoch(self, epoch):
        self.model.train()
        metrics = {'total': 0.0}
        y_true, y_pred = [], []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            inputs = []
            if 'feat_v' in batch: inputs.append(batch['feat_v'].to(self.device))
            if 'feat_e' in batch: inputs.append(batch['feat_e'].to(self.device))
            if 'feat_c' in batch: inputs.append(batch['feat_c'].to(self.device))

            if self.task == 'cls':
                labels = batch["label"].to(self.device).long()
            else:
                labels = batch["label"].to(self.device).float()

            self.optimizer.zero_grad()
            outputs = self.model(*inputs)

            if self.task == 'cls':
                loss = self.criterion(outputs["y_pred"], labels)
                preds = outputs["y_pred"].argmax(dim=1)
            else:
                preds = outputs["y_pred"].squeeze(-1)
                loss = self.criterion(preds, labels)

            loss.backward()
            self.optimizer.step()

            metrics['total'] += loss.item() * labels.size(0)
            y_true.extend(labels.cpu().detach().numpy().tolist())
            y_pred.extend(preds.cpu().detach().numpy().tolist())

        avg_loss = metrics['total'] / len(self.train_loader.dataset)
        self.writer.add_scalar('Loss/Total', avg_loss, epoch)

        if self.task == 'cls':
            acc, f1 = compute_metrics_cognitive(y_true, y_pred)
            self.writer.add_scalar('Metrics/Train_ACC', acc, epoch)
            return avg_loss, acc
        else:
            ccc, _, mae = compute_metrics_regression(y_true, y_pred)
            self.writer.add_scalar('Metrics/Train_CCC', ccc, epoch)
            return avg_loss, ccc

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        y_true, y_pred = [], []

        for batch in self.val_loader:
            inputs = []
            if 'feat_v' in batch: inputs.append(batch['feat_v'].to(self.device))
            if 'feat_e' in batch: inputs.append(batch['feat_e'].to(self.device))
            if 'feat_c' in batch: inputs.append(batch['feat_c'].to(self.device))

            if self.task == 'cls':
                labels = batch["label"].to(self.device).long()
            else:
                labels = batch["label"].to(self.device).float()

            outputs = self.model(*inputs)

            if self.task == 'cls':
                preds = outputs["y_pred"].argmax(dim=1)
            else:
                preds = outputs["y_pred"].squeeze(-1)

            y_true.extend(labels.cpu().detach().numpy().tolist())
            y_pred.extend(preds.cpu().detach().numpy().tolist())

        if self.task == 'cls':
            acc, f1 = compute_metrics_cognitive(y_true, y_pred)
            return acc, f1
        else:
            ccc, rmse, mae = compute_metrics_regression(y_true, y_pred)
            return ccc, mae, rmse

    def run(self):
        best_stats = {}
        for epoch in range(1, self.args.epochs + 1):
            train_loss, train_metric = self.train_one_epoch(epoch)

            if self.task == 'cls':
                val_acc, val_f1 = self.evaluate()
                primary_metric = val_acc
                self.logger.info(
                    f"Epoch {epoch:02d}: Tr_Loss={train_loss:.4f}, Tr_ACC={train_metric:.4f} | Val_ACC={val_acc:.4f}, Val_F1={val_f1:.4f}")
            else:
                val_ccc, val_mae, val_rmse = self.evaluate()
                primary_metric = val_ccc
                self.logger.info(
                    f"Epoch {epoch:02d}: Tr_Loss={train_loss:.4f}, Tr_CCC={train_metric:.4f} | Val_CCC={val_ccc:.4f}, Val_MAE={val_mae:.4f}, Val_RMSE={val_rmse:.4f}")

            self.scheduler.step()
            self.writer.add_scalar('Metrics/Val_Primary', primary_metric, epoch)

            if primary_metric > self.best_metric:
                self.best_metric = primary_metric
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best_model.ckpt"))
                if self.task == 'cls':
                    best_stats = {"acc": round(val_acc, 4), "f1": round(val_f1, 4)}
                else:
                    best_stats = {"ccc": round(val_ccc, 4), "mae": round(val_mae, 4), "rmse": round(val_rmse, 4)}
                self.logger.info("--> Checkpoint updated.")

            if self.early_stopper(primary_metric):
                self.logger.info("Early stopping triggered.")
                break

        return best_stats