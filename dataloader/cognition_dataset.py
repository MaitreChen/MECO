import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class MECO_SI_COG_Dataset(Dataset):
    """
    Dataset for Subject-Independent (SI) multimodal Cognitive tasks (MMSE).
    Calculates normalization statistics strictly on the provided sample_indices
    to prevent data leakage during Cross-Validation.
    Supports both Classification ('cls') and Regression ('reg') tasks.
    """

    def __init__(self,
                 data_source,  # Pre-loaded full dataset dictionary
                 task="cls",  # 'cls' for binary screening, 'reg' for MMSE score
                 feat_type_video=None,
                 feat_type_eeg=None,
                 feat_type_ecg=None,
                 sample_indices=None,
                 s_mean_v=None, s_std_v=None,
                 s_mean_e=None, s_std_e=None,
                 s_mean_c=None, s_std_c=None):

        self.task = task
        self.label_type = "label_mmse"

        # 1. Parse indices
        total_samples = len(data_source[self.label_type])
        self.indices = sample_indices if sample_indices is not None else list(range(total_samples))

        # Extract labels only for the given indices
        self.all_label = [data_source[self.label_type][i] for i in self.indices]

        # 2. Format feature types as lists
        self.feat_type_video = [feat_type_video] if isinstance(feat_type_video, str) else (feat_type_video or [])
        self.feat_type_eeg = [feat_type_eeg] if isinstance(feat_type_eeg, str) else (feat_type_eeg or [])
        self.feat_type_ecg = [feat_type_ecg] if isinstance(feat_type_ecg, str) else (feat_type_ecg or [])

        # 3. Reference the pre-loaded feature dictionaries
        self.all_feats_v = data_source.get('video', {})
        self.all_feats_e = data_source.get('eeg', {})
        self.all_feats_c = data_source.get('ecg', {})

        # 4. Initialize normalization statistics based ONLY on self.indices
        self.s_mean_v_dict, self.s_std_v_dict = self._init_modality_stats(self.feat_type_video, self.all_feats_v,
                                                                          s_mean_v,
                                                                          s_std_v) if self.feat_type_video else ({}, {})
        self.s_mean_e_dict, self.s_std_e_dict = self._init_modality_stats(self.feat_type_eeg, self.all_feats_e,
                                                                          s_mean_e,
                                                                          s_std_e) if self.feat_type_eeg else ({}, {})
        self.s_mean_c_dict, self.s_std_c_dict = self._init_modality_stats(self.feat_type_ecg, self.all_feats_c,
                                                                          s_mean_c,
                                                                          s_std_c) if self.feat_type_ecg else ({}, {})

    def _init_modality_stats(self, feat_types, all_feats_dict, s_mean_input, s_std_input):
        mean_dict, std_dict = {}, {}
        for ft in feat_types:
            if ft not in all_feats_dict: continue
            feats_subset = [all_feats_dict[ft][i] for i in self.indices]
            feats_tensor = self._to_tensor(feats_subset)

            if s_mean_input is None or s_std_input is None or (
                    isinstance(s_mean_input, dict) and ft not in s_mean_input):
                mean, std = self._compute_stats(feats_tensor)
            else:
                mean = s_mean_input[ft].clone() if isinstance(s_mean_input, dict) else s_mean_input.clone()
                std = s_std_input[ft].clone() if isinstance(s_std_input, dict) else s_std_input.clone()

            std[std == 0] = 1e-6
            mean_dict[ft] = mean
            std_dict[ft] = std
        return mean_dict, std_dict

    def _to_tensor(self, feats_list):
        if isinstance(feats_list, list):
            if torch.is_tensor(feats_list[0]):
                feats_tensor = torch.stack(feats_list).float()
            else:
                feats_tensor = torch.tensor(np.array(feats_list), dtype=torch.float32)
        else:
            feats_tensor = torch.tensor(feats_list, dtype=torch.float32)

        if feats_tensor.dim() == 4 and feats_tensor.shape[1] == 1:
            feats_tensor = feats_tensor.squeeze(1)
        return feats_tensor

    def _compute_stats(self, feats_tensor):
        if feats_tensor.dim() == 3:
            return feats_tensor.mean(dim=[0, 1]), feats_tensor.std(dim=[0, 1])
        elif feats_tensor.dim() == 2:
            return feats_tensor.mean(dim=0), feats_tensor.std(dim=0)
        return feats_tensor.mean(), feats_tensor.std()

    def _process_modality_features(self, feat_types, all_feats_dict, mean_dict, std_dict, global_idx):
        if not feat_types or not all_feats_dict: return None
        normed_list = []
        for ft in feat_types:
            if ft not in all_feats_dict: continue
            feat = all_feats_dict[ft][global_idx]
            feat = torch.tensor(feat, dtype=torch.float32).clone().detach() if not torch.is_tensor(
                feat) else feat.clone().detach().float()
            if feat.dim() == 3 and feat.shape[0] == 1: feat = feat.squeeze(0)
            feat_norm = (feat - mean_dict[ft]) / std_dict[ft]
            normed_list.append(feat_norm)
        if not normed_list: return None
        return torch.cat(normed_list, dim=-1) if len(normed_list) > 1 else normed_list[0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        global_idx = self.indices[idx]

        combined_feat_v = self._process_modality_features(self.feat_type_video, self.all_feats_v, self.s_mean_v_dict,
                                                          self.s_std_v_dict, global_idx)
        combined_feat_e = self._process_modality_features(self.feat_type_eeg, self.all_feats_e, self.s_mean_e_dict,
                                                          self.s_std_e_dict, global_idx)
        combined_feat_c = self._process_modality_features(self.feat_type_ecg, self.all_feats_c, self.s_mean_c_dict,
                                                          self.s_std_c_dict, global_idx)

        raw_label = self.all_label[idx]

        # Task mapping: Binary Classification or Regression
        if self.task == 'cls':
            # > 26 is Healthy (0), <= 26 is MCI (1)
            mapped_label = 0 if raw_label > 26 else 1
            label = mapped_label
        else:
            # Direct continuous float for regression
            label = float(raw_label)

        item = {"label": label}
        if combined_feat_v is not None: item["feat_v"] = combined_feat_v
        if combined_feat_e is not None: item["feat_e"] = combined_feat_e
        if combined_feat_c is not None: item["feat_c"] = combined_feat_c

        return item