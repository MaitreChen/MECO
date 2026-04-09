import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataloader.emotion_dataset import MECO_SD_CLS_CachedDataset
from utils.general import get_logger
from utils.metrics import compute_metrics_emotion, EarlyStopping
from models.baseline import FusionNet


class Trainer:
    """
    Handles the training, validation, and testing pipeline for multimodal classification.
    Supports dynamic modality routing (V, E, C).
    """

    def __init__(self, args, train_data, val_data, test_data, label_type, save_dir, val_sub):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.logger = get_logger(self.save_dir)
        self.writer = SummaryWriter(self.save_dir)
        self.val_sub = val_sub
        num_classes = self.args.num_classes

        # 1. Modality Configuration Gatekeeper
        self.used_modalities = getattr(args, "used_modalities", ["video", "eeg", "ecg"])
        self.feat_type_video = getattr(args, 'feature_type_video',
                                       'deep_feature') if 'video' in self.used_modalities else None
        self.feat_type_eeg = getattr(args, 'feature_type_eeg',
                                     'eeg_de_feats') if 'eeg' in self.used_modalities else None
        self.feat_type_ecg = getattr(args, 'feature_type_ecg',
                                     'ecg_hfd_feats') if 'ecg' in self.used_modalities else None

        self._log_modality_configuration()

        # 2. Dataset Instantiation
        self.train_dataset = MECO_SD_CLS_CachedDataset(
            data_source=train_data, label_type=label_type,
            feat_type_video=self.feat_type_video, feat_type_eeg=self.feat_type_eeg, feat_type_ecg=self.feat_type_ecg,
            num_classes=num_classes
        )

        # Retrieve normalization statistics computed on the training set
        train_mean_v, train_std_v = self.train_dataset.s_mean_v_dict, self.train_dataset.s_std_v_dict
        train_mean_e, train_std_e = self.train_dataset.s_mean_e_dict, self.train_dataset.s_std_e_dict
        train_mean_c, train_std_c = self.train_dataset.s_mean_c_dict, self.train_dataset.s_std_c_dict

        self.val_dataset = MECO_SD_CLS_CachedDataset(
            data_source=val_data, label_type=label_type,
            feat_type_video=self.feat_type_video, feat_type_eeg=self.feat_type_eeg, feat_type_ecg=self.feat_type_ecg,
            num_classes=num_classes,
            s_mean_v=train_mean_v, s_std_v=train_std_v, s_mean_e=train_mean_e, s_std_e=train_std_e,
            s_mean_c=train_mean_c, s_std_c=train_std_c
        )

        self.test_dataset = MECO_SD_CLS_CachedDataset(
            data_source=test_data, label_type=label_type,
            feat_type_video=self.feat_type_video, feat_type_eeg=self.feat_type_eeg, feat_type_ecg=self.feat_type_ecg,
            num_classes=num_classes,
            s_mean_v=train_mean_v, s_std_v=train_std_v, s_mean_e=train_mean_e, s_std_e=train_std_e,
            s_mean_c=train_mean_c, s_std_c=train_std_c
        )

        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=args.batch_size, shuffle=False)

        # 3. Dynamic Model Architecture Construction
        active_feature_types = []
        if self.feat_type_video: active_feature_types.append(self.feat_type_video)
        if self.feat_type_eeg: active_feature_types.append(self.feat_type_eeg)
        if self.feat_type_ecg: active_feature_types.append(self.feat_type_ecg)

        self.model = FusionNet(
            feature_types=active_feature_types,
            num_classes=num_classes,
            use_gru=True,
            dropout_rate=0.3
        ).to(self.device)

        # 4. Optimization Setup
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=1e-5)
        self.best_uar = 0.0
        self.early_stopper = EarlyStopping(patience=args.patience, mode="max")

    def _log_modality_configuration(self):
        """Persists the experimental configuration to the log file."""
        self.logger.info("=== Modalities & Features Configuration ===")
        if self.feat_type_video: self.logger.info(f"[*] Video: {self.feat_type_video}")
        if self.feat_type_eeg:   self.logger.info(f"[*] EEG  : {self.feat_type_eeg}")
        if self.feat_type_ecg:   self.logger.info(f"[*] ECG  : {self.feat_type_ecg}")
        self.logger.info("===========================================")

    def train_one_epoch(self, epoch, dataloader):
        self.model.train()
        metrics = {'total': 0.0}
        y_true, y_pred = [], []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            # Dynamically assemble inputs respecting the V -> E -> C order
            inputs = []
            if 'feat_v' in batch: inputs.append(batch['feat_v'].to(self.device))
            if 'feat_e' in batch: inputs.append(batch['feat_e'].to(self.device))
            if 'feat_c' in batch: inputs.append(batch['feat_c'].to(self.device))

            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(*inputs)
            loss = self.criterion(outputs["y_pred"], labels)

            loss.backward()
            self.optimizer.step()

            metrics['total'] += loss.item() * labels.size(0)
            preds = outputs["y_pred"].argmax(dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

            pbar.set_postfix({'L_total': f"{loss.item():.3f}"})

        avg_loss = metrics['total'] / len(dataloader.dataset)
        uar, _, _ = compute_metrics_emotion(y_true, y_pred)

        self.writer.add_scalar('Loss/Total', avg_loss, epoch)
        self.writer.add_scalar('Metrics/Train_UAR', uar, epoch)

        return avg_loss, uar

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        y_true, y_pred = [], []

        for batch in dataloader:
            inputs = []
            if 'feat_v' in batch: inputs.append(batch['feat_v'].to(self.device))
            if 'feat_e' in batch: inputs.append(batch['feat_e'].to(self.device))
            if 'feat_c' in batch: inputs.append(batch['feat_c'].to(self.device))

            labels = batch["label"].to(self.device)
            outputs = self.model(*inputs)
            preds = outputs["y_pred"].argmax(dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

        uar, war, f1 = compute_metrics_emotion(y_true, y_pred)
        return uar, war, f1, y_true, y_pred

    def run(self):
        """Executes the complete training loop with validation and early stopping."""
        best_stats = {"uar": 0.0, "war": 0.0, "f1": 0.0}

        for epoch in range(1, self.args.epochs + 1):
            train_loss, train_uar = self.train_one_epoch(epoch, self.train_loader)
            val_uar, val_war, val_f1, _, _ = self.evaluate(self.val_loader)

            self.scheduler.step()

            self.logger.info(
                f"Epoch {epoch:02d}: "
                f"Tr_Loss={train_loss:.4f}, Tr_UAR={train_uar:.4f} | "
                f"Val_UAR={val_uar:.4f}, Val_WAR={val_war:.4f}, Val_F1={val_f1:.4f}"
            )

            self.writer.add_scalar('Metrics/Val_UAR', val_uar, epoch)

            # Checkpoint saving and test evaluation upon improvement
            if val_uar > self.best_uar:
                self.best_uar = val_uar
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best_model.ckpt"))

                test_uar, test_war, test_f1, _, _ = self.evaluate(self.test_loader)
                best_stats = {"uar": round(test_uar, 4), "war": round(test_war, 4), "f1": round(test_f1, 4)}

                self.logger.info(
                    f"--> Checkpoint updated. Test UAR={test_uar:.4f}, WAR={test_war:.4f}, F1={test_f1:.4f}")

            if self.early_stopper(val_uar):
                self.logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

        self.logger.info(
            f"Final Best Test UAR={best_stats['uar']:.4f}, WAR={best_stats['war']:.4f}, F1={best_stats['f1']:.4f}")
        return best_stats
