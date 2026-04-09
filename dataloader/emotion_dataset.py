import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class MECO_SD_CLS_CachedDataset(Dataset):
    """
    Dataset for Subject-Dependent multimodal (Video + EEG + ECG) classification.
    Supports flexible modality combinations. Missing or excluded modalities are gracefully handled.
    Features are independently Z-score normalized and concatenated along the feature dimension (dim=-1).
    """

    def __init__(self,
                 data_source,
                 label_type,
                 feat_type_video,  # e.g., 'deep_feature' or ['au_openface', 'deep_feature'] or None
                 feat_type_eeg,  # e.g., 'eeg_de_feats' or None
                 feat_type_ecg,  # e.g., 'ecg_sampen_feats' or None
                 num_classes=5,
                 s_mean_v=None, s_std_v=None,
                 s_mean_e=None, s_std_e=None,
                 s_mean_c=None, s_std_c=None):

        # 1. Load data
        if isinstance(data_source, (str, os.PathLike)):
            with open(data_source, "rb") as f:
                cached_data = pickle.load(f)
        else:
            cached_data = data_source

        self.subject_id = cached_data.get('subject', 'Unknown')
        self.num_classes = num_classes
        self.label_type = label_type
        self.all_label = cached_data[self.label_type]

        # 2. Format feature types as lists; default to empty list if None
        self.feat_type_video = [feat_type_video] if isinstance(feat_type_video, str) else (feat_type_video or [])
        self.feat_type_eeg = [feat_type_eeg] if isinstance(feat_type_eeg, str) else (feat_type_eeg or [])
        self.feat_type_ecg = [feat_type_ecg] if isinstance(feat_type_ecg, str) else (feat_type_ecg or [])

        # 3. Extract features only if they exist in the cached data
        self.all_feats_v = {ft: cached_data[ft] for ft in self.feat_type_video if ft in cached_data}
        self.all_feats_e = {ft: cached_data[ft] for ft in self.feat_type_eeg if ft in cached_data}
        self.all_feats_c = {ft: cached_data[ft] for ft in self.feat_type_ecg if ft in cached_data}

        # 4. Initialize normalization statistics (mean/std) only for active modalities
        self.s_mean_v_dict, self.s_std_v_dict = self._init_modality_stats(self.feat_type_video, self.all_feats_v,
                                                                          s_mean_v, s_std_v) if self.all_feats_v else (
            {}, {})
        self.s_mean_e_dict, self.s_std_e_dict = self._init_modality_stats(self.feat_type_eeg, self.all_feats_e,
                                                                          s_mean_e, s_std_e) if self.all_feats_e else (
            {}, {})
        self.s_mean_c_dict, self.s_std_c_dict = self._init_modality_stats(self.feat_type_ecg, self.all_feats_c,
                                                                          s_mean_c, s_std_c) if self.all_feats_c else (
            {}, {})

        # 5. Verify length alignment only for active modalities
        if self.feat_type_video and self.feat_type_video[0] in self.all_feats_v:
            assert len(self.all_feats_v[self.feat_type_video[0]]) == len(
                self.all_label), "Video feature length mismatch!"
        if self.feat_type_eeg and self.feat_type_eeg[0] in self.all_feats_e:
            assert len(self.all_feats_e[self.feat_type_eeg[0]]) == len(self.all_label), "EEG feature length mismatch!"
        if self.feat_type_ecg and self.feat_type_ecg[0] in self.all_feats_c:
            assert len(self.all_feats_c[self.feat_type_ecg[0]]) == len(self.all_label), "ECG feature length mismatch!"

    def _init_modality_stats(self, feat_types, all_feats_dict, s_mean_input, s_std_input):
        """Calculate or load mean and std for all features within a modality."""
        mean_dict, std_dict = {}, {}
        for ft in feat_types:
            feats_tensor = self._to_tensor(all_feats_dict[ft])

            # Compute stats if not provided
            if s_mean_input is None or s_std_input is None or (
                    isinstance(s_mean_input, dict) and ft not in s_mean_input):
                mean, std = self._compute_stats(feats_tensor)
            else:
                # Support both dict and tensor inputs
                mean = s_mean_input[ft].clone() if isinstance(s_mean_input, dict) else s_mean_input.clone()
                std = s_std_input[ft].clone() if isinstance(s_std_input, dict) else s_std_input.clone()

            std[std == 0] = 1e-6  # Prevent division by zero
            mean_dict[ft] = mean
            std_dict[ft] = std

        return mean_dict, std_dict

    def _to_tensor(self, feats_list):
        """Convert feature lists to Tensors, squeezing redundant dimensions."""
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
        """Compute mean and std based on tensor dimensionality."""
        if feats_tensor.dim() == 3:
            return feats_tensor.mean(dim=[0, 1]), feats_tensor.std(dim=[0, 1])
        elif feats_tensor.dim() == 2:
            return feats_tensor.mean(dim=0), feats_tensor.std(dim=0)
        return feats_tensor.mean(), feats_tensor.std()

    def _process_modality_features(self, feat_types, all_feats_dict, mean_dict, std_dict, idx):
        """Extract, normalize, and concatenate multiple features for a single modality."""
        # FIX: Return None immediately if modality is inactive
        if not feat_types or not all_feats_dict:
            return None

        normed_list = []
        for ft in feat_types:
            if ft not in all_feats_dict:
                continue

            feat = all_feats_dict[ft][idx]
            feat = torch.tensor(feat, dtype=torch.float32).clone().detach() if not torch.is_tensor(
                feat) else feat.clone().detach().float()

            if feat.dim() == 3 and feat.shape[0] == 1:
                feat = feat.squeeze(0)

            # Independent Z-score normalization
            feat_norm = (feat - mean_dict[ft]) / std_dict[ft]
            normed_list.append(feat_norm)

        # Concatenate along the last dimension if multiple features exist
        if not normed_list:
            return None
        return torch.cat(normed_list, dim=-1) if len(normed_list) > 1 else normed_list[0]

    def __len__(self):
        return len(self.all_label)

    def __getitem__(self, idx):
        # 1. Process features; returns None if modality is missing/inactive
        combined_feat_v = self._process_modality_features(self.feat_type_video, self.all_feats_v, self.s_mean_v_dict,
                                                          self.s_std_v_dict, idx)
        combined_feat_e = self._process_modality_features(self.feat_type_eeg, self.all_feats_e, self.s_mean_e_dict,
                                                          self.s_std_e_dict, idx)
        combined_feat_c = self._process_modality_features(self.feat_type_ecg, self.all_feats_c, self.s_mean_c_dict,
                                                          self.s_std_c_dict, idx)

        # 2. Label mapping
        raw_label = self.all_label[idx]
        label_val = raw_label.item() if torch.is_tensor(raw_label) else raw_label
        mapped_label = label_val

        if self.num_classes == 5:
            mapping = {1: 3, 2: 1, 4: 2, 5: 4}
            mapped_label = mapping.get(label_val, label_val)
        elif self.num_classes == 3:
            mapped_label = 0 if label_val in [0, 2, 4] else (1 if label_val in [1, 3] else 2)
        elif self.num_classes == 2:
            mapped_label = 0 if label_val in [0, 2, 4] else 1

        label = mapped_label if self.label_type == 'label_sa_class' else label_val

        # 3. Dynamically build the return dictionary (Only include active modalities)
        item = {"label": label}
        if combined_feat_v is not None: item["feat_v"] = combined_feat_v
        if combined_feat_e is not None: item["feat_e"] = combined_feat_e
        if combined_feat_c is not None: item["feat_c"] = combined_feat_c

        return item


class MECO_SI_CLS_Dataset(Dataset):
    """
    Dataset for Subject-Independent (SI) multimodal classification.
    Calculates normalization statistics strictly on the provided sample_indices
    to prevent data leakage during Cross-Validation.
    """

    def __init__(self,
                 data_source,  # A dictionary containing pre-loaded full dataset
                 label_type,
                 feat_type_video=None,
                 feat_type_eeg=None,
                 feat_type_ecg=None,
                 sample_indices=None,
                 num_classes=5,
                 s_mean_v=None, s_std_v=None,
                 s_mean_e=None, s_std_e=None,
                 s_mean_c=None, s_std_c=None):

        self.num_classes = num_classes
        self.label_type = label_type

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
        """Calculate mean/std using ONLY the data specified by self.indices."""
        mean_dict, std_dict = {}, {}
        for ft in feat_types:
            if ft not in all_feats_dict:
                continue

            # Extract subset based on train indices to prevent data leakage
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
        """Extract, normalize, and concatenate features using the global index."""
        if not feat_types or not all_feats_dict:
            return None

        normed_list = []
        for ft in feat_types:
            if ft not in all_feats_dict:
                continue

            feat = all_feats_dict[ft][global_idx]
            feat = torch.tensor(feat, dtype=torch.float32).clone().detach() if not torch.is_tensor(
                feat) else feat.clone().detach().float()

            if feat.dim() == 3 and feat.shape[0] == 1:
                feat = feat.squeeze(0)

            feat_norm = (feat - mean_dict[ft]) / std_dict[ft]
            normed_list.append(feat_norm)

        if not normed_list:
            return None
        return torch.cat(normed_list, dim=-1) if len(normed_list) > 1 else normed_list[0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Map local dataset index to the global data_source index
        global_idx = self.indices[idx]

        combined_feat_v = self._process_modality_features(self.feat_type_video, self.all_feats_v, self.s_mean_v_dict,
                                                          self.s_std_v_dict, global_idx)
        combined_feat_e = self._process_modality_features(self.feat_type_eeg, self.all_feats_e, self.s_mean_e_dict,
                                                          self.s_std_e_dict, global_idx)
        combined_feat_c = self._process_modality_features(self.feat_type_ecg, self.all_feats_c, self.s_mean_c_dict,
                                                          self.s_std_c_dict, global_idx)

        raw_label = self.all_label[idx]  # self.all_label is already filtered by indices
        label_val = raw_label.item() if torch.is_tensor(raw_label) else raw_label
        mapped_label = label_val

        # Label Mapping
        if self.num_classes == 5:
            mapping = {1: 3, 2: 1, 4: 2, 5: 4}
            mapped_label = mapping.get(label_val, label_val)
        elif self.num_classes == 3:
            mapped_label = 0 if label_val in [0, 2, 4] else (1 if label_val in [1, 3] else 2)
        elif self.num_classes == 2:
            mapped_label = 0 if label_val in [0, 2, 4] else 1

        label = mapped_label if self.label_type == 'label_sa_class' else label_val

        item = {"label": label}
        if combined_feat_v is not None: item["feat_v"] = combined_feat_v
        if combined_feat_e is not None: item["feat_e"] = combined_feat_e
        if combined_feat_c is not None: item["feat_c"] = combined_feat_c

        return item


class MECO_SD_REG_CachedDataset(Dataset):
    """
    Dataset for Subject-Dependent multimodal (Video + EEG + ECG) Regression.
    Supports flexible modality combinations and dynamic feature concatenation.
    """

    def __init__(self,
                 data_source,
                 label_type,  # e.g., 'label_arousal' or 'label_valence'
                 feat_type_video=None,
                 feat_type_eeg=None,
                 feat_type_ecg=None,
                 s_mean_v=None, s_std_v=None,
                 s_mean_e=None, s_std_e=None,
                 s_mean_c=None, s_std_c=None):

        # 1. Load data
        if isinstance(data_source, (str, os.PathLike)):
            with open(data_source, "rb") as f:
                cached_data = pickle.load(f)
        else:
            cached_data = data_source

        self.subject_id = cached_data.get('subject', 'Unknown')
        self.label_type = label_type
        self.all_label = cached_data[self.label_type]

        # 2. Format feature types as lists
        self.feat_type_video = [feat_type_video] if isinstance(feat_type_video, str) else (feat_type_video or [])
        self.feat_type_eeg = [feat_type_eeg] if isinstance(feat_type_eeg, str) else (feat_type_eeg or [])
        self.feat_type_ecg = [feat_type_ecg] if isinstance(feat_type_ecg, str) else (feat_type_ecg or [])

        # 3. Extract features only if they exist
        self.all_feats_v = {ft: cached_data[ft] for ft in self.feat_type_video if ft in cached_data}
        self.all_feats_e = {ft: cached_data[ft] for ft in self.feat_type_eeg if ft in cached_data}
        self.all_feats_c = {ft: cached_data[ft] for ft in self.feat_type_ecg if ft in cached_data}

        # 4. Initialize normalization statistics only for active modalities
        self.s_mean_v_dict, self.s_std_v_dict = self._init_modality_stats(self.feat_type_video, self.all_feats_v,
                                                                          s_mean_v, s_std_v) if self.all_feats_v else (
            {}, {})
        self.s_mean_e_dict, self.s_std_e_dict = self._init_modality_stats(self.feat_type_eeg, self.all_feats_e,
                                                                          s_mean_e, s_std_e) if self.all_feats_e else (
            {}, {})
        self.s_mean_c_dict, self.s_std_c_dict = self._init_modality_stats(self.feat_type_ecg, self.all_feats_c,
                                                                          s_mean_c, s_std_c) if self.all_feats_c else (
            {}, {})

        # 5. Verify length alignment
        if self.feat_type_video and self.feat_type_video[0] in self.all_feats_v:
            assert len(self.all_feats_v[self.feat_type_video[0]]) == len(
                self.all_label), "Video feature length mismatch!"
        if self.feat_type_eeg and self.feat_type_eeg[0] in self.all_feats_e:
            assert len(self.all_feats_e[self.feat_type_eeg[0]]) == len(self.all_label), "EEG feature length mismatch!"
        if self.feat_type_ecg and self.feat_type_ecg[0] in self.all_feats_c:
            assert len(self.all_feats_c[self.feat_type_ecg[0]]) == len(self.all_label), "ECG feature length mismatch!"

    def _init_modality_stats(self, feat_types, all_feats_dict, s_mean_input, s_std_input):
        mean_dict, std_dict = {}, {}
        for ft in feat_types:
            feats_tensor = self._to_tensor(all_feats_dict[ft])
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

    def _process_modality_features(self, feat_types, all_feats_dict, mean_dict, std_dict, idx):
        if not feat_types or not all_feats_dict: return None
        normed_list = []
        for ft in feat_types:
            if ft not in all_feats_dict: continue
            feat = all_feats_dict[ft][idx]
            feat = torch.tensor(feat, dtype=torch.float32).clone().detach() if not torch.is_tensor(
                feat) else feat.clone().detach().float()
            if feat.dim() == 3 and feat.shape[0] == 1: feat = feat.squeeze(0)
            feat_norm = (feat - mean_dict[ft]) / std_dict[ft]
            normed_list.append(feat_norm)
        if not normed_list: return None
        return torch.cat(normed_list, dim=-1) if len(normed_list) > 1 else normed_list[0]

    def __len__(self):
        return len(self.all_label)

    def __getitem__(self, idx):
        combined_feat_v = self._process_modality_features(self.feat_type_video, self.all_feats_v, self.s_mean_v_dict,
                                                          self.s_std_v_dict, idx)
        combined_feat_e = self._process_modality_features(self.feat_type_eeg, self.all_feats_e, self.s_mean_e_dict,
                                                          self.s_std_e_dict, idx)
        combined_feat_c = self._process_modality_features(self.feat_type_ecg, self.all_feats_c, self.s_mean_c_dict,
                                                          self.s_std_c_dict, idx)

        # 回归任务直接使用原始浮点标签，无类别映射
        raw_label = self.all_label[idx]
        label = float(raw_label.item() if torch.is_tensor(raw_label) else raw_label)

        item = {"label": label}
        if combined_feat_v is not None: item["feat_v"] = combined_feat_v
        if combined_feat_e is not None: item["feat_e"] = combined_feat_e
        if combined_feat_c is not None: item["feat_c"] = combined_feat_c
        return item


class MECO_SI_REG_Dataset(Dataset):
    """
    Dataset for Subject-Independent (SI) multimodal Regression.
    Calculates normalization statistics strictly on the provided sample_indices
    to prevent data leakage during Cross-Validation.
    Outputs continuous labels (e.g., Arousal/Valence).
    """

    def __init__(self,
                 data_source,  # Pre-loaded full dataset dictionary
                 label_type,  # 'label_arousal' or 'label_valence'
                 feat_type_video=None,
                 feat_type_eeg=None,
                 feat_type_ecg=None,
                 sample_indices=None,
                 s_mean_v=None, s_std_v=None,
                 s_mean_e=None, s_std_e=None,
                 s_mean_c=None, s_std_c=None):

        self.label_type = label_type

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
        """Calculate mean/std using ONLY the data specified by self.indices."""
        mean_dict, std_dict = {}, {}
        for ft in feat_types:
            if ft not in all_feats_dict: continue
            # Extract subset based on train indices to prevent data leakage
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
        # Map local dataset index to the global data_source index
        global_idx = self.indices[idx]

        combined_feat_v = self._process_modality_features(self.feat_type_video, self.all_feats_v, self.s_mean_v_dict,
                                                          self.s_std_v_dict, global_idx)
        combined_feat_e = self._process_modality_features(self.feat_type_eeg, self.all_feats_e, self.s_mean_e_dict,
                                                          self.s_std_e_dict, global_idx)
        combined_feat_c = self._process_modality_features(self.feat_type_ecg, self.all_feats_c, self.s_mean_c_dict,
                                                          self.s_std_c_dict, global_idx)

        # Regression label parsing (continuous float)
        raw_label = self.all_label[idx]
        label = float(raw_label.item() if torch.is_tensor(raw_label) else raw_label)

        item = {"label": label}
        if combined_feat_v is not None: item["feat_v"] = combined_feat_v
        if combined_feat_e is not None: item["feat_e"] = combined_feat_e
        if combined_feat_c is not None: item["feat_c"] = combined_feat_c

        return item
