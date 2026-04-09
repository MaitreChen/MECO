import os
import json
import argparse
import gc
from datetime import datetime

import torch
import numpy as np

# Project internal imports
from utils.general import set_seed
from utils.io import format_feature_name
from dataloader.data_utils import merge_multimodal_pkls
from engine.trainer_sd_reg import TrainerSDReg


def save_regression_results(save_dir, subject_id, best_metrics):
    """Saves the final evaluation metrics for a single subject to a JSON file."""
    results = {
        "subject": subject_id,
        "Test": {
            "CCC": f"{best_metrics['ccc']:.4f}",
            "MAE": f"{best_metrics['mae']:.4f}",
            "RMSE": f"{best_metrics['rmse']:.4f}"
        }
    }
    save_path = os.path.join(save_dir, f"{subject_id}_results_summary.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)


def print_experiment_header(used_modalities, label_type, feat_v_name, feat_e_name, feat_c_name):
    """Prints a highly visible summary of the active experimental setup."""
    print("\n" + "✨" * 20)
    print(f"🚀 [Active Modalities Configuration - SD Regression ({label_type})]")
    if 'video' in used_modalities: print(f"   🎥 Video -> Features: {feat_v_name}")
    if 'eeg' in used_modalities:   print(f"   🧠 EEG   -> Features: {feat_e_name}")
    if 'ecg' in used_modalities:   print(f"   🫀 ECG   -> Features: {feat_c_name}")
    print("✨" * 20 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Subject-Dependent Multimodal Emotion Regression")
    parser.add_argument("--config_file", default="configs/emotion_sd/full_reg.json", type=str)
    args = parser.parse_args()

    # Merge JSON config into argparse namespace
    with open(args.config_file, "r") as f:
        cfg = json.load(f)
    args = argparse.Namespace(**vars(args), **cfg)

    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # 1. Modality Control & Validation
    used_modalities = getattr(args, "used_modalities", ["video", "eeg", "ecg"])
    label_type = getattr(args, "label_type", "label_valence")  # Default to valence

    # Use standardized Subject IDs from S1 to S42
    all_subject_ids = [f'S{i}' for i in range(1, 43)]
    print(f"[INFO] Total Subjects found: {len(all_subject_ids)}")

    feat_v_name = getattr(args, 'feature_type_video', 'deep_feature') if 'video' in used_modalities else None
    feat_e_name = getattr(args, 'feature_type_eeg', 'eeg_de_feats') if 'eeg' in used_modalities else None
    feat_c_name = getattr(args, 'feature_type_ecg', 'ecg_time_feats') if 'ecg' in used_modalities else None

    print_experiment_header(used_modalities, label_type, feat_v_name, feat_e_name, feat_c_name)

    # 2. Workspace Initialization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"EMO-SD-REG-{label_type}-{format_feature_name(feat_v_name)}-{format_feature_name(feat_e_name)}-{format_feature_name(feat_c_name)}-{timestamp}"
    master_log_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(master_log_dir, exist_ok=True)

    results = {}

    # 3. Main Subject Loop
    for subject_id in all_subject_ids:
        print(f"\n{'=' * 20} Processing Subject: {subject_id} {'=' * 20}")

        def get_paths(phase):
            paths = {}
            if "video" in used_modalities: paths["video"] = os.path.join(args.sd_data_dir_video,
                                                                         f"{subject_id}_{phase}.pkl")
            if "eeg" in used_modalities: paths["eeg"] = os.path.join(args.sd_data_dir_eeg, f"{subject_id}_{phase}.pkl")
            if "ecg" in used_modalities: paths["ecg"] = os.path.join(args.sd_data_dir_ecg, f"{subject_id}_{phase}.pkl")
            return paths

        train_paths = get_paths("train")
        valid_paths = get_paths("valid")
        test_paths = get_paths("test")

        # Skip subject if required modality data is missing
        if not all(os.path.exists(p) for p in train_paths.values()):
            print(
                f"[WARN] Subject {subject_id} is missing required data for modalities: {used_modalities}. Skipping...")
            continue

        train_data = merge_multimodal_pkls(train_paths, label_type)
        valid_data = merge_multimodal_pkls(valid_paths, label_type)
        test_data = merge_multimodal_pkls(test_paths, label_type)

        subject_save_dir = os.path.join(master_log_dir, subject_id)
        os.makedirs(subject_save_dir, exist_ok=True)

        trainer = TrainerSDReg(
            args=args,
            train_data=train_data,
            val_data=valid_data,
            test_data=test_data,
            label_type=label_type,
            save_dir=subject_save_dir,
            val_sub=subject_id,
        )

        best_metrics = trainer.run()
        results[subject_id] = best_metrics
        save_regression_results(subject_save_dir, subject_id, best_metrics)

        # Free GPU memory before processing the next subject
        del trainer
        torch.cuda.empty_cache()
        gc.collect()

    # 4. Final Aggregation
    if results:
        all_cccs = [res["ccc"] for res in results.values()]
        all_maes = [res["mae"] for res in results.values()]
        all_rmses = [res["rmse"] for res in results.values()]

        print(f"\n[FINAL SUMMARY] Evaluated on {len(results)} subjects ({label_type}).")
        print(f"Average CCC  : {np.mean(all_cccs):.4f} ± {np.std(all_cccs):.4f}")
        print(f"Average MAE  : {np.mean(all_maes):.4f} ± {np.std(all_maes):.4f}")
        print(f"Average RMSE : {np.mean(all_rmses):.4f} ± {np.std(all_rmses):.4f}")

        final_summary_path = os.path.join(master_log_dir, "Final_SD_REG_Results.json")
        with open(final_summary_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"[INFO] Detailed summary saved to: {final_summary_path}")
    else:
        print("[ERROR] No subjects were successfully processed.")


if __name__ == "__main__":
    main()