import torch
import warnings
from pathlib import Path
from classes.generic_lightning_module import GenericLightningNetwork
import openvino as ov
import torch

# WIP

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
DIRECTORY_NAME = rf"./tb_logs/models"
BASE_MODEL_NAME = DIRECTORY_NAME + "/optimized_model"
weights_path = Path(BASE_MODEL_NAME + ".pt")

# Paths where ONNX and OpenVINO IR models will be stored.
onnx_path = weights_path.with_suffix('.onnx')
if not onnx_path.parent.exists():
    onnx_path.parent.mkdir()
ir_path = onnx_path.with_suffix(".xml")

# CODICE TIPO INFERENZA
# create model object
model = GenericLightningNetwork(
        parsed_layers=architecture,
        input_channels=in_channels,
        #input_height=256,
        #input_width=256,
        num_classes=num_classes,
        learning_rate=lr,
        model_parameters=model_parameters,
    )
)
# read state dict, use map_location argument to avoid a situation where weights are saved in cuda (which may not be unavailable on the system)
state_dict = torch.load(weights_path, map_location='cpu')
# load state dict to model
model.load_state_dict(state_dict)
# switch model from training to inference mode
model.eval()
print("Loaded Pytorch model.")

# CONVERT PYTORCH MODEL TO ONNX
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    if not onnx_path.exists():
        dummy_input = torch.randn(1, 4, IMAGE_HEIGHT, IMAGE_WIDTH)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
        )
        print(f"ONNX model exported to {onnx_path}.")
    else:
        print(f"ONNX model {onnx_path} already exists.")

# CONVERT ONNX MODEL TO OPENVINO IR FORMAT
if not ir_path.exists():
    print("Exporting ONNX model to IR... This may take a few minutes.")
    ov_model = ov.convert_model(onnx_path)
    ov.save_model(ov_model, ir_path)
else:
    print(f"IR model {ir_path} already exists.")
