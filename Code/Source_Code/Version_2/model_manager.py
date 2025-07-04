import os
import json
import torch

MODELS_DIR = "models"
METADATA_FILE = os.path.join(MODELS_DIR, "metadata.json")

def ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)

def load_metadata():
    if not os.path.exists(METADATA_FILE):
        return []
    with open(METADATA_FILE, 'r') as f:
        return json.load(f)

def save_metadata(metadata):
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)

def store_model(model, training_profit, validation_profit, testing_profit, performance, model_name=None):
    ensure_dirs()
    metadata = load_metadata()
    if model_name is None:
        model_name = f"agent_{len(metadata)+1}.pt"
    model_path = os.path.join(MODELS_DIR, model_name)

    torch.save(model.state_dict(), model_path)

    entry = {
        "model_path": model_name,  # store relative path for portability
        "training_profit_change": training_profit,
        "validation_profit_change": validation_profit,
        "testing_profit_change": testing_profit,
        "performance": performance
    }
    metadata.append(entry)
    save_metadata(metadata)
    print(f"Stored model: {model_name}")

def load_model(index, model_class):
    metadata = load_metadata()
    if index < 0 or index >= len(metadata):
        raise IndexError(f"Model index {index} out of range")
    entry = metadata[index]
    model_path = os.path.join(MODELS_DIR, entry["model_path"])

    model = model_class()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def list_performance():
    metadata = load_metadata()
    return [entry["performance"] for entry in metadata]

def display_info(index):
    metadata = load_metadata()
    if index < 0 or index >= len(metadata):
        print(f"Model index {index} out of range.")
        return
    entry = metadata[index]
    print(f"Model #{index+1}:")
    print(f"  Path: {entry['model_path']}")
    print(f"  Performance: {entry['performance']:.4f}")
    print(f"  Training profit length: {len(entry['training_profit_change'])}")
    print(f"  Validation profit length: {len(entry['validation_profit_change'])}")
    print(f"  Testing profit length: {len(entry['testing_profit_change'])}")

def delete_model(index):
    metadata = load_metadata()
    if index < 0 or index >= len(metadata):
        print(f"Model index {index} out of range.")
        return
    entry = metadata.pop(index)
    model_path = os.path.join(MODELS_DIR, entry["model_path"])
    try:
        os.remove(model_path)
        print(f"Deleted model file: {model_path}")
    except OSError as e:
        print(f"Error deleting model file: {e}")
    save_metadata(metadata)
