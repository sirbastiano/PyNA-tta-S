from pynattas.utils.utils import safe_import

# Network-related imports
safe_import("pynattas.builder.generic_lightning_module", [
    "GenericLightningNetwork",
    "GenericLightningNetwork_Custom",
    "GenericOD_YOLOv3",
    "GenericOD_YOLOv3_SmallObjects"
])

safe_import("pynattas.builder.generic_network", ["GenericNetwork"])

# Early stopping imports
safe_import("pynattas.builder.my_early_stopping", ["EarlyStopping", "TrainEarlyStopping"])