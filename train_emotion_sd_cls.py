import os
import json
import argparse
import gc
from datetime import datetime

import torch
import numpy as np

# Project internal imports
from utils.general import set_seed
from utils.io import save_subject_results, format_feature_name
from dataloader.data_utils import merge_multimodal_pkls
from engine.trainer_sd_cls import Trainer


def print_experiment_header(used_modalities, feat_v_name, feat_e_name, feat_c_name):
    """Prints a highly visible summary of the active experimental setup."""
    print("\n" + "✨" * 20)
    print("🚀 [Active Modalities Configuration]")
    if 'video' in used_modalities: print(f"   🎥 Video -> Features: {feat_v_name}")
    if 'eeg' in used_modalities:   print(f"   🧠 EEG   -> Features: {feat_e_name}")
    if 'ecg' in used_modalities:   print(f"   🫀 ECG   -> Features: {feat_c_name}")
    print("✨" * 20 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Subject-Dependent Multimodal Emotion Classification")
    parser.add_argument("--config_file", default="configs/emotion_sd/full_cls.json", type=str)
    args = parser.parse_args()

    # Merge JSON config into argparse namespace
    with open(args.config_file, "r") as f:
        cfg = json.load(f)
    args = argparse.Namespace(**vars(args), **cfg)

    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # 1. Modality Control & Validation
    used_modalities = getattr(args, "used_modalities", ["video", "eeg", "ecg"])

    # Optional: Read from file or define explicitly
    all_subject_ids = [f'S{i}' for i in range(1, 43)]
    print(f"[INFO] Total Subjects found: {len(all_subject_ids)}")

    feat_v_name = getattr(args, 'feature_type_video', 'deep_feature') if 'video' in used_modalities else None
    feat_e_name = getattr(args, 'feature_type_eeg', 'eeg_de_feats') if 'eeg' in used_modalities else None
    feat_c_name = getattr(args, 'feature_type_ecg', 'ecg_hfd_feats') if 'ecg' in used_modalities else None

    print_experiment_header(used_modalities, feat_v_name, feat_e_name, feat_c_name)

    # 2. Workspace Initialization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"EMO-SD-CLS-{format_feature_name(feat_v_name)}-{format_feature_name(feat_e_name)}-{format_feature_name(feat_c_name)}-nc{args.num_classes}-{timestamp}"
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

        # Skip subject if required modality data is missing in the training split
        if not all(os.path.exists(p) for p in train_paths.values()):
            print(
                f"[WARN] Subject {subject_id} is missing required data for modalities: {used_modalities}. Skipping...")
            continue

        # Load and merge multimodal structures
        train_data = merge_multimodal_pkls(train_paths, args.label_type)
        valid_data = merge_multimodal_pkls(valid_paths, args.label_type)
        test_data = merge_multimodal_pkls(test_paths, args.label_type)

        subject_save_dir = os.path.join(master_log_dir, subject_id)
        os.makedirs(subject_save_dir, exist_ok=True)

        # Instantiate trainer for current subject
        trainer = Trainer(
            args=args,
            train_data=train_data,
            val_data=valid_data,
            test_data=test_data,
            label_type=args.label_type,
            save_dir=subject_save_dir,
            val_sub=subject_id,
        )

        best_metrics = trainer.run()
        results[subject_id] = best_metrics
        save_subject_results(subject_save_dir, subject_id, best_metrics)

        # Free GPU memory before next subject
        del trainer
        torch.cuda.empty_cache()
        gc.collect()

    # 4. Final Aggregation
    if results:
        all_uars = [res["uar"] for res in results.values()]
        all_wars = [res["war"] for res in results.values()]
        all_f1s = [res["f1"] for res in results.values()]

        print(f"\n[FINAL SUMMARY] Evaluated on {len(results)} subjects.")
        print(f"Average UAR : {np.mean(all_uars):.4f} ± {np.std(all_uars):.4f}")
        print(f"Average WAR : {np.mean(all_wars):.4f} ± {np.std(all_wars):.4f}")
        print(f"Average F1  : {np.mean(all_f1s):.4f} ± {np.std(all_f1s):.4f}")

        final_summary_path = os.path.join(master_log_dir, "Final_SD_Results.json")
        with open(final_summary_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"[INFO] Detailed summary saved to: {final_summary_path}")
    else:
        print("[ERROR] No subjects were successfully processed. Check your data paths and modality settings.")


if __name__ == "__main__":
    main()
