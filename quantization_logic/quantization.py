import os
import torch
import onnx
import numpy as np
import tensorflow as tf
from onnx_tf.backend import prepare
from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import tempfile

# ------------------------------
# 1. Dataset Preparation (Fixed for Calibration)
# ------------------------------
def prepare_dataset(dataset_path="person_detection_validation"):
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset path {dataset_path} does not exist. Using a dummy dataset for calibration.")
        # Create a dummy dataset with random data for calibration
        return DummyDataset()
        
    try:
        transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        full_dataset = datasets.ImageFolder(dataset_path, transform=transform)
        val_size = int(0.1 * len(full_dataset))
        test_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size - test_size

        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size], generator=generator
        )

        return val_dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return DummyDataset()

class DummyDataset:
    """Dummy dataset for when real dataset isn't available"""
    def __init__(self, size=100):
        self.size = size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Return random tensor of shape [1, 96, 96] (grayscale image)
        return torch.rand(1, 96, 96), 0  # Random image, label 0

def representative_dataset_gen(val_dataset):
    for i in range(min(100, len(val_dataset))):
        try:
            image, _ = val_dataset[i]
            image = image.numpy()
            image = np.expand_dims(image, axis=0)
            yield [image.astype(np.float32)]
        except Exception as e:
            # If there's an error with a particular sample, just use random data
            random_image = np.random.rand(1, 1, 96, 96).astype(np.float32)
            yield [random_image]

# ------------------------------
# 2. Conversion Functions
# ------------------------------
def convert_pth_to_onnx(pth_model_path, onnx_model_path, input_shape=(1, 1, 96, 96)):
    try:
        # Load the full model, not just state dict
        model = torch.load(pth_model_path, map_location=torch.device('cpu'))
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape)
        
        # Export to ONNX
        torch.onnx.export(
            model, dummy_input, onnx_model_path,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=13
        )
        print(f"ONNX model saved to {onnx_model_path}")
        return True
    except Exception as e:
        print(f"Error in ONNX conversion: {e}")
        return False

def convert_onnx_to_tflite(onnx_model_path, tflite_output_path, val_dataset, quantization_type="int8"):
    try:
        # Load ONNX and convert to TensorFlow SavedModel
        onnx_model = onnx.load(onnx_model_path)
        tf_rep = prepare(onnx_model)

        saved_model_dir = tempfile.mkdtemp()
        tf_rep.export_graph(saved_model_dir)

        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if quantization_type == "int8":
            converter.representative_dataset = lambda: representative_dataset_gen(val_dataset)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        elif quantization_type == "float16":
            converter.target_spec.supported_types = [tf.float16]
        elif quantization_type == "dynamic":
            # No representative dataset required for dynamic range quantization
            pass
        else:
            print(f"Unknown quantization type '{quantization_type}'. Falling back to int8.")
            converter.representative_dataset = lambda: representative_dataset_gen(val_dataset)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

        tflite_model = converter.convert()

        with open(tflite_output_path, "wb") as f:
            f.write(tflite_model)
        print(f"TFLite quantized model saved to {tflite_output_path}")

        import shutil
        shutil.rmtree(saved_model_dir)

        return True
    except Exception as e:
        print(f"Error in TFLite conversion: {e}")
        return False

# ------------------------------
# 3. Master Function to Call from Frontend
# ------------------------------
def quantize_model(pth_model_path, tflite_output_path, dataset_path="person_detection_validation", quantization_type="int8"):
    try:
        val_dataset = prepare_dataset(dataset_path)
        onnx_model_path = os.path.join(tempfile.gettempdir(), "temp_model.onnx")

        if not convert_pth_to_onnx(pth_model_path, onnx_model_path):
            print("ONNX conversion failed, aborting quantization")
            return False

        success = convert_onnx_to_tflite(onnx_model_path, tflite_output_path, val_dataset, quantization_type)

        if os.path.exists(onnx_model_path):
            os.remove(onnx_model_path)

        return success
    except Exception as e:
        print(f"Error during quantization: {e}")
        return False

def get_tflite_model_size(tflite_model_path):
    """Returns the TFLite model size in MB."""
    if os.path.exists(tflite_model_path):
        size_mb = os.path.getsize(tflite_model_path) / (1024 * 1024)
        return round(size_mb, 2)
    else:
        print(f"File {tflite_model_path} does not exist.")
        return 0
