import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataloader.emotion_dataset import MECO_SD_REG_CachedDataset
from utils.general import get_logger
from utils.metrics import compute_metrics_regression, EarlyStopping
from models.baseline import FusionNet


class TrainerSDReg:
    """
    Trainer for Subject-Dependent Multimodal Regression tasks.
    """

    def __init__(self, args, train_data, val_data, test_data, label_type, save_dir, val_sub):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.logger = get_logger(self.save_dir)
        self.writer = SummaryWriter(self.save_dir)
        self.val_sub = val_sub

        # Regression tasks require a single output dimension
        num_classes = 1

        # 1. Modality Gatekeeper
        self.used_modalities = getattr(args, "used_modalities", ["video", "eeg", "ecg"])
        self.feat_type_video = getattr(args, 'feature_type_video',
                                       'deep_feature') if 'video' in self.used_modalities else None
        self.feat_type_eeg = getattr(args, 'feature_type_eeg',
                                     'eeg_de_feats') if 'eeg' in self.used_modalities else None
        self.feat_type_ecg = getattr(args, 'feature_type_ecg',
                                     'ecg_time_feats') if 'ecg' in self.used_modalities else None

        self._log_modality_configuration()

        # 2. Dataset Setup
        # Initialize train dataset to compute normalization statistics
        self.train_dataset = MECO_SD_REG_CachedDataset(
            data_source=train_data, label_type=label_type,
            feat_type_video=self.feat_type_video, feat_type_eeg=self.feat_type_eeg, feat_type_ecg=self.feat_type_ecg
        )

        # Retrieve computed statistics
        train_mean_v, train_std_v = self.train_dataset.s_mean_v_dict, self.train_dataset.s_std_v_dict
        train_mean_e, train_std_e = self.train_dataset.s_mean_e_dict, self.train_dataset.s_std_e_dict
        train_mean_c, train_std_c = self.train_dataset.s_mean_c_dict, self.train_dataset.s_std_c_dict

        # Apply train statistics to validation and test datasets
        self.val_dataset = MECO_SD_REG_CachedDataset(
            data_source=val_data, label_type=label_type,
            feat_type_video=self.feat_type_video, feat_type_eeg=self.feat_type_eeg, feat_type_ecg=self.feat_type_ecg,
            s_mean_v=train_mean_v, s_std_v=train_std_v, s_mean_e=train_mean_e, s_std_e=train_std_e,
            s_mean_c=train_mean_c, s_std_c=train_std_c
        )

        self.test_dataset = MECO_SD_REG_CachedDataset(
            data_source=test_data, label_type=label_type,
            feat_type_video=self.feat_type_video, feat_type_eeg=self.feat_type_eeg, feat_type_ecg=self.feat_type_ecg,
            s_mean_v=train_mean_v, s_std_v=train_std_v, s_mean_e=train_mean_e, s_std_e=train_std_e,
            s_mean_c=train_mean_c, s_std_c=train_std_c
        )

        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=args.batch_size, shuffle=False)

        # 3. Dynamic Model Setup
        active_feature_types = []
        if self.feat_type_video: active_feature_types.append(self.feat_type_video)
        if self.feat_type_eeg: active_feature_types.append(self.feat_type_eeg)
        if self.feat_type_ecg: active_feature_types.append(self.feat_type_ecg)

        self.model = FusionNet(feature_types=active_feature_types, num_classes=num_classes, use_gru=True).to(
            self.device)

        # 4. Optimization & Metrics Setup
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=1e-5)

        self.best_ccc = -1.0  # Higher CCC is better
        self.early_stopper = EarlyStopping(patience=args.patience, mode="max")

    def _log_modality_configuration(self):
        """Logs the active modalities and feature types to the log file."""
        self.logger.info("=== Modalities & Features Configuration (Regression) ===")
        if self.feat_type_video: self.logger.info(f"[*] Video: {self.feat_type_video}")
        if self.feat_type_eeg:   self.logger.info(f"[*] EEG  : {self.feat_type_eeg}")
        if self.feat_type_ecg:   self.logger.info(f"[*] ECG  : {self.feat_type_ecg}")
        self.logger.info("======================================================")

    def train_one_epoch(self, epoch):
        self.model.train()
        metrics = {'total': 0.0}
        y_true, y_pred = [], []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            # Dynamically assemble inputs respecting the strict order
            inputs = []
            if 'feat_v' in batch: inputs.append(batch['feat_v'].to(self.device))
            if 'feat_e' in batch: inputs.append(batch['feat_e'].to(self.device))
            if 'feat_c' in batch: inputs.append(batch['feat_c'].to(self.device))

            labels = batch["label"].to(self.device).float()

            self.optimizer.zero_grad()
            outputs = self.model(*inputs)

            # Squeeze output from shape (B, 1) to (B,) for regression loss
            preds = outputs["y_pred"].squeeze(-1)
            loss = self.criterion(preds, labels)

            loss.backward()
            self.optimizer.step()

            metrics['total'] += loss.item() * labels.size(0)
            y_true.extend(labels.cpu().detach().numpy().tolist())
            y_pred.extend(preds.cpu().detach().numpy().tolist())

            pbar.set_postfix({'L_MSE': f"{loss.item():.4f}"})

        avg_loss = metrics['total'] / len(self.train_loader.dataset)
        ccc, _, _ = compute_metrics_regression(y_true, y_pred)

        self.writer.add_scalar('Loss/Total', avg_loss, epoch)
        self.writer.add_scalar('Metrics/Train_CCC', ccc, epoch)

        return avg_loss, ccc

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        y_true, y_pred = [], []

        for batch in dataloader:
            inputs = []
            if 'feat_v' in batch: inputs.append(batch['feat_v'].to(self.device))
            if 'feat_e' in batch: inputs.append(batch['feat_e'].to(self.device))
            if 'feat_c' in batch: inputs.append(batch['feat_c'].to(self.device))

            labels = batch["label"].to(self.device).float()
            outputs = self.model(*inputs)
            preds = outputs["y_pred"].squeeze(-1)

            y_true.extend(labels.cpu().detach().numpy().tolist())
            y_pred.extend(preds.cpu().detach().numpy().tolist())

        ccc, rmse, mae = compute_metrics_regression(y_true, y_pred)
        return ccc, mae, rmse

    def run(self):
        """Executes the complete training loop with validation and early stopping."""
        best_stats = {"ccc": -1.0, "mae": 999.0, "rmse": 999.0}

        for epoch in range(1, self.args.epochs + 1):
            train_loss, train_ccc = self.train_one_epoch(epoch)
            val_ccc, val_mae, val_rmse = self.evaluate(self.val_loader)
            self.scheduler.step()

            self.logger.info(
                f"Epoch {epoch:02d}: "
                f"Tr_Loss={train_loss:.4f}, Tr_CCC={train_ccc:.4f} | "
                f"Val_CCC={val_ccc:.4f}, Val_MAE={val_mae:.4f}, Val_RMSE={val_rmse:.4f}"
            )

            self.writer.add_scalar('Metrics/Val_CCC', val_ccc, epoch)

            # Checkpoint saving and test evaluation upon improvement
            if val_ccc > self.best_ccc:
                self.best_ccc = val_ccc
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best_model.ckpt"))

                test_ccc, test_mae, test_rmse = self.evaluate(self.test_loader)
                best_stats = {"ccc": round(test_ccc, 4), "mae": round(test_mae, 4), "rmse": round(test_rmse, 4)}
                self.logger.info(
                    f"--> Checkpoint updated. Test CCC={test_ccc:.4f}, MAE={test_mae:.4f}, RMSE={test_rmse:.4f}")

            if self.early_stopper(val_ccc):
                self.logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

        self.logger.info(
            f"Final Best Test CCC={best_stats['ccc']:.4f}, MAE={best_stats['mae']:.4f}, RMSE={best_stats['rmse']:.4f}")
        return best_stats