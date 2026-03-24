import pickle
import os
import torch


def _sanitize_path(path):
    """Replace characters that are invalid in filenames."""
    for ch in [':', ',', '<', '>', "'", '"', '(', ')']:
        path = path.replace(ch, '_')
    # Collapse multiple underscores
    while '__' in path:
        path = path.replace('__', '_')
    return path


def save_model_state_dict(destination_path, model):

    destination_path = _sanitize_path(destination_path)
    if not destination_path.endswith(".pth"):
        destination_path += ".pth"

    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    torch.save(model.state_dict(), destination_path)
    print(f"Model state_dict saved to {destination_path}.")


def load_model_state_dict(path, model):

    path = _sanitize_path(path)
    if not path.endswith(".pth"):
        path += ".pth"

    if not os.path.exists(path):
        return None

    model.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
    return model


# Keep old names as aliases for compatibility
def save_model_state_dict_to_gcs(bucket_name, destination_blob_name, model):
    path = os.path.join(bucket_name, destination_blob_name)
    return save_model_state_dict(path, model)


def load_model_state_dict_from_gcs(bucket_name, blob_name, model):
    path = os.path.join(bucket_name, blob_name)
    return load_model_state_dict(path, model)