import os
import json
import argparse
import pickle
import gc
from datetime import datetime

import torch
import numpy as np

from utils.general import set_seed
from utils.io import load_subject_list
from engine.trainer_si_cog import TrainerSICog


def save_cog_results(save_dir, fold_id, best_metrics, task='reg'):
    results = {"fold": fold_id, "Validation": best_metrics}
    save_path = os.path.join(save_dir, f"{fold_id}_results_summary.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Subject-Independent Cognitive Regression")
    parser.add_argument("--config_file", default="configs/cognition_si/cog_reg.json", type=str)
    parser.add_argument("--task", default="reg", type=str, help="Enforce regression task")
    parser.add_argument("--pair", type=str, choices=["V", "E", "C", "VE", "VC", "EC", "VEC"], default='VEC')
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        cfg = json.load(f)
    args = argparse.Namespace(**vars(args), **cfg)

    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # Convert '--pair' logic cleanly into our standard 'used_modalities'
    used_modalities = []
    if "V" in args.pair: used_modalities.append("video")
    if "E" in args.pair: used_modalities.append("eeg")
    if "C" in args.pair: used_modalities.append("ecg")
    setattr(args, "used_modalities", used_modalities)

    # 1. Remapped MMSE Dictionary (S1 to S42 strictly based on the mapping provided)
    subject_mmse_dict = {
        "S1": 23, "S2": 24, "S3": 29, "S4": 29, "S5": 12, "S6": 14, "S7": 25, "S8": 30, "S9": 27, "S10": 30,
        "S11": 26, "S12": 23, "S13": 25, "S14": 28, "S15": 26, "S16": 29, "S17": 27, "S18": 25, "S19": 27, "S20": 28,
        "S21": 30, "S22": 19, "S23": 17, "S24": 30, "S25": 29, "S26": 28, "S27": 29, "S28": 30, "S29": 29, "S30": 24,
        "S31": 30, "S32": 30, "S33": 29, "S34": 29, "S35": 29, "S36": 23, "S37": 28, "S38": 30, "S39": 27, "S40": 29,
        "S41": 29, "S42": 26
    }
    all_subjects = [f'S{i}' for i in range(1, 43)]

    feat_v_name = getattr(args, 'feature_type_video', 'deep_feature') if 'video' in used_modalities else None
    feat_e_name = getattr(args, 'feature_type_eeg', 'eeg_de_feats') if 'eeg' in used_modalities else None
    feat_c_name = getattr(args, 'feature_type_ecg', 'ecg_time_feats') if 'ecg' in used_modalities else None

    print("\n" + "✨" * 20)
    print(f"🚀 [SI Cognitive Regression (MMSE Score) - Modality: {args.pair}]")
    print("✨" * 20 + "\n")

    # 2. Pre-load Full Dataset
    fmt_lst = lambda x: [x] if isinstance(x, str) else (x or [])
    active_feats_v = fmt_lst(feat_v_name)
    active_feats_e = fmt_lst(feat_e_name)
    active_feats_c = fmt_lst(feat_c_name)

    full_data_dict = {
        "label_mmse": [], "subject_ids": [],
        "video": {f: [] for f in active_feats_v},
        "eeg": {f: [] for f in active_feats_e},
        "ecg": {f: [] for f in active_feats_c}
    }

    subject_ranges = {}
    current_idx = 0

    print(f"[INFO] Pre-loading features into memory...")
    for s in all_subjects:
        if s not in subject_mmse_dict: continue
        sub_mmse_score = subject_mmse_dict[s]

        pkl_path_v = args.cached_v_pkl_format.format(data_root=args.data_root_video,
                                                     subject=s) if 'video' in used_modalities else None
        pkl_path_e = args.cached_e_pkl_format.format(data_root=args.data_root_eeg,
                                                     subject=s) if 'eeg' in used_modalities else None
        pkl_path_c = args.cached_c_pkl_format.format(data_root=args.data_root_ecg,
                                                     subject=s) if 'ecg' in used_modalities else None

        paths_to_check = [p for p in [pkl_path_v, pkl_path_e, pkl_path_c] if p is not None]
        if not all(os.path.exists(p) for p in paths_to_check): continue

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

            ref_feats = sess_v if 'video' in used_modalities else (sess_e if 'eeg' in used_modalities else sess_c)
            ref_f_name = active_feats_v[0] if active_feats_v else (
                active_feats_e[0] if active_feats_e else active_feats_c[0])
            num_samples = len(ref_feats[ref_f_name])

            all_lengths = []
            for f in active_feats_v: all_lengths.append(len(sess_v[f]))
            for f in active_feats_e: all_lengths.append(len(sess_e[f]))
            for f in active_feats_c: all_lengths.append(len(sess_c[f]))

            if any(l != num_samples for l in all_lengths): continue

            for f in active_feats_v: full_data_dict["video"][f].extend(sess_v[f])
            for f in active_feats_e: full_data_dict["eeg"][f].extend(sess_e[f])
            for f in active_feats_c: full_data_dict["ecg"][f].extend(sess_c[f])

            full_data_dict["label_mmse"].extend([sub_mmse_score] * num_samples)
            full_data_dict["subject_ids"].extend([s] * num_samples)

            subject_ranges.setdefault(s, [])
            subject_ranges[s].extend(range(current_idx, current_idx + num_samples))
            current_idx += num_samples

    # 3. Execution
    run_name = f"Cog-REG-5CV-{args.pair}-nc1-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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

        if not train_idx or not val_idx: continue

        fold_save_dir = os.path.join(master_log_dir, f"Fold_{fold_id}")
        trainer = TrainerSICog(
            args=args, full_data_dict=full_data_dict,
            train_indices=train_idx, val_indices=val_idx,
            save_dir=fold_save_dir, val_sub=f"Fold{fold_id}"
        )

        best_metrics = trainer.run()
        results[f"Fold{fold_id}"] = best_metrics
        save_cog_results(fold_save_dir, f"Fold{fold_id}", best_metrics, task='reg')

        del trainer
        torch.cuda.empty_cache()
        gc.collect()

    if results:
        print("\n" + "=" * 40)
        for m_name in ["ccc", "mae", "rmse"]:
            vals = [r[m_name] for r in results.values()]
            print(f"[FINAL] 5-Fold CV Average {m_name.upper()} ({args.pair}): {np.mean(vals):.4f} ± {np.std(vals):.4f}")

        final_summary_path = os.path.join(master_log_dir, "Final_SI_COG_REG_Results.json")
        with open(final_summary_path, "w") as f:
            json.dump(results, f, indent=4)
    else:
        print("[ERROR] No folds were successfully executed.")


if __name__ == "__main__":
    main()