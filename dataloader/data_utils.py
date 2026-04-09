import os
import pickle


def merge_multimodal_pkls(paths_dict, label_type):
    """
    Merges multimodal pickle files into a single dictionary.
    Gracefully handles missing modalities to support ablation studies.

    Args:
        paths_dict (dict): Maps modality names to file paths (e.g., {'video': path}).
        label_type (str): The key used to extract the target label.
    """
    merged_data = {"modalities": []}

    for m_type, path in paths_dict.items():
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)

            # Initialize shared metadata from the first valid modality
            if "subject" not in merged_data:
                merged_data["subject"] = data.get("subject", "Unknown")
                merged_data[label_type] = data[label_type]

            # Copy features and register the active modality
            for k, v in data.items():
                if k not in ["subject", label_type]:
                    merged_data[k] = v
            merged_data["modalities"].append(m_type)

    return merged_data