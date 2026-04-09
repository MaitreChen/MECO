import os
import json
import ast


def load_subject_list(txt_path):
    """Safely loads a list of subjects from a text file."""
    with open(txt_path, "r") as f:
        return ast.literal_eval(f.read().strip())


def save_subject_results(save_dir, subject_id, best_metrics):
    """Dumps subject-specific evaluation metrics to a JSON file."""
    results = {
        "subject": subject_id,
        "Test": {
            "UAR": f"{best_metrics['uar']:.4f}",
            "WAR": f"{best_metrics['war']:.4f}",
            "F1": f"{best_metrics['f1']:.4f}"
        }
    }
    save_path = os.path.join(save_dir, f"{subject_id}_results_summary.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[INFO] Results saved to {save_path}")


def format_feature_name(feat):
    """Formats feature names for clean directory naming."""
    if not feat:
        return "None"
    return "-".join(feat) if isinstance(feat, list) else str(feat)
