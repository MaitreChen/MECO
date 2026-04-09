import os
import json
import argparse
import pickle
import gc
from datetime import datetime

import torch
import numpy as np

# Project internal imports
from utils.general import set_seed
from utils.io import format_feature_name, load_subject_list
from engine.trainer_si_reg import TrainerSIReg


def save_regression_results(save_dir, fold_id, best_metrics):
    """Saves the final evaluation metrics for a single fold to a JSON file."""
    results = {
        "fold": fold_id,
        "Validation": {
            "CCC": f"{best_metrics['ccc']:.4f}",
            "MAE": f"{best_metrics['mae']:.4f}",
            "RMSE": f"{best_metrics['rmse']:.4f}"
        }
    }
    save_path = os.path.join(save_dir, f"{fold_id}_results_summary.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)


def print_experiment_header(used_modalities, label_type, feat_v_name, feat_e_name, feat_c_name):
    print("\n" + "✨" * 20)
    print(f"🚀 [SI Active Modalities Configuration - Regression ({label_type})]")
    if 'video' in used_modalities: print(f"   🎥 Video -> Features: {feat_v_name}")
    if 'eeg' in used_modalities:   print(f"   🧠 EEG   -> Features: {feat_e_name}")
    if 'ecg' in used_modalities:   print(f"   🫀 ECG   -> Features: {feat_c_name}")
    print("✨" * 20 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Subject-Independent Multimodal Emotion Regression")
    parser.add_argument("--config_file", default="configs/emotion_si/full_reg.json", type=str)
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        cfg = json.load(f)
    args = argparse.Namespace(**vars(args), **cfg)

    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # 1. Modality & Label Control
    used_modalities = getattr(args, "used_modalities", ["video", "eeg", "ecg"])
    label_type = getattr(args, "label_type", "label_arousal")  # Default to Arousal

    feat_v_name = getattr(args, 'feature_type_video', 'deep_feature') if 'video' in used_modalities else None
    feat_e_name = getattr(args, 'feature_type_eeg', 'eeg_de_feats') if 'eeg' in used_modalities else None
    feat_c_name = getattr(args, 'feature_type_ecg', 'ecg_time_feats') if 'ecg' in used_modalities else None

    # Format into lists for dynamic extraction
    fmt_lst = lambda x: [x] if isinstance(x, str) else (x or [])
    active_feats_v = fmt_lst(feat_v_name)
    active_feats_e = fmt_lst(feat_e_name)
    active_feats_c = fmt_lst(feat_c_name)

    print_experiment_header(used_modalities, label_type, feat_v_name, feat_e_name, feat_c_name)

    # 2. Pre-load Full Dataset
    all_subjects = [f'S{i}' for i in range(1, 43)]
    print(f"[INFO] Pre-loading features into memory...")

    full_data_dict = {
        label_type: [],
        "subject_ids": [],
        "video": {f: [] for f in active_feats_v},
        "eeg": {f: [] for f in active_feats_e},
        "ecg": {f: [] for f in active_feats_c}
    }

    subject_ranges = {}
    current_idx = 0

    for s in all_subjects:
        pkl_path_v = args.cached_v_pkl_format.format(data_root=args.data_root_video,
                                                     subject=s) if 'video' in used_modalities else None
        pkl_path_e = args.cached_e_pkl_format.format(data_root=args.data_root_eeg,
                                                     subject=s) if 'eeg' in used_modalities else None
        pkl_path_c = args.cached_c_pkl_format.format(data_root=args.data_root_ecg,
                                                     subject=s) if 'ecg' in used_modalities else None

        paths_to_check = [p for p in [pkl_path_v, pkl_path_e, pkl_path_c] if p is not None]
        if not all(os.path.exists(p) for p in paths_to_check):
            print(f"[WARNING] Missing modality file for {s}. Skipping.")
            continue

        data_v = pickle.load(open(pkl_path_v, "rb")) if pkl_path_v else {}
        data_e = pickle.load(open(pkl_path_e, "rb")) if pkl_path_e else {}
        data_c = pickle.load(open(pkl_path_c, "rb")) if pkl_path_c else {}

        for sess in ["session1", "session2", "session3"]:
            sess_v = data_v.get(sess, {})
            sess_e = data_e.get(sess, {})
            sess_c = data_c.get(sess, {})

            if any(f not in sess_v for f in active_feats_v) or \
                    any(f not in sess_e for f in active_feats_e) or \
                    any(f not in sess_c for f in active_feats_c):
                continue

            labels = sess_v.get(label_type) or sess_e.get(label_type) or sess_c.get(label_type)
            if labels is None: continue

            num_samples = len(labels)

            all_feature_lengths = []
            for f in active_feats_v: all_feature_lengths.append(len(sess_v[f]))
            for f in active_feats_e: all_feature_lengths.append(len(sess_e[f]))
            for f in active_feats_c: all_feature_lengths.append(len(sess_c[f]))

            if any(l != num_samples for l in all_feature_lengths):
                print(f"[WARNING] Unmatched sample length for {s} {sess}. Skipping.")
                continue

            for f in active_feats_v: full_data_dict["video"][f].extend(sess_v[f])
            for f in active_feats_e: full_data_dict["eeg"][f].extend(sess_e[f])
            for f in active_feats_c: full_data_dict["ecg"][f].extend(sess_c[f])

            full_data_dict[label_type].extend(labels)
            full_data_dict["subject_ids"].extend([s] * num_samples)

            subject_ranges.setdefault(s, [])
            subject_ranges[s].extend(range(current_idx, current_idx + num_samples))
            current_idx += num_samples

    # 3. Setup Fold Cross-Validation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"EMO-SI-REG-{label_type}-{format_feature_name(feat_v_name)}-{format_feature_name(feat_e_name)}-{format_feature_name(feat_c_name)}-{timestamp}"
    master_log_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(master_log_dir, exist_ok=True)
    results = {}

    for fold_id in range(1, 6):
        print(f"\n{'=' * 15} Starting Fold {fold_id} {'=' * 15}")
        fold_dir = os.path.join("splits", f"fold{fold_id}")
        train_subs = load_subject_list(os.path.join(fold_dir, "train_subjects.txt"))
        test_subs = load_subject_list(os.path.join(fold_dir, "test_subjects.txt"))

        train_idx = [i for s in train_subs if f"S{s}" in subject_ranges for i in subject_ranges[f"S{s}"]]
        val_idx = [i for s in test_subs if f"S{s}" in subject_ranges for i in subject_ranges[f"S{s}"]]

        if not train_idx or not val_idx:
            print(f"[WARNING] Empty train/val split for Fold {fold_id}. Skipping.")
            continue

        fold_save_dir = os.path.join(master_log_dir, f"Fold_{fold_id}")
        trainer = TrainerSIReg(
            args=args,
            full_data_dict=full_data_dict,
            train_indices=train_idx,
            val_indices=val_idx,
            label_type=label_type,
            save_dir=fold_save_dir,
            val_sub=f"Fold{fold_id}"
        )

        best_metrics = trainer.run()
        results[f"Fold{fold_id}"] = best_metrics
        save_regression_results(fold_save_dir, f"Fold{fold_id}", best_metrics)

        del trainer
        torch.cuda.empty_cache()
        gc.collect()

    # 4. Final Aggregation
    if results:
        print("\n" + "=" * 40)
        for m_name in ["ccc", "mae", "rmse"]:
            vals = [r[m_name] for r in results.values()]
            print(f"[FINAL] 5-Fold CV Average {m_name.upper()}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

        final_summary_path = os.path.join(master_log_dir, "Final_SI_REG_Results.json")
        with open(final_summary_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"[INFO] Detailed summary saved to: {final_summary_path}")
    else:
        print("[ERROR] No folds were successfully executed.")


if __name__ == "__main__":
    main()