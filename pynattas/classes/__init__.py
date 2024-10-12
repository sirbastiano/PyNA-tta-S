import logging

# Set up logging configuration for better debugging and control over message levels
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Helper function to handle imports and log failures
def safe_import(module_name, class_names):
    """Attempts to import classes from a module and logs any ImportErrors.
    
    Parameters:
    module_name (str): The name of the module from which to import.
    class_names (list): A list of class names (as strings) to import from the module.
    
    Returns:
    None
    """
    try:
        module = __import__(module_name, globals(), locals(), class_names, 0)
        globals().update({cls: getattr(module, cls) for cls in class_names})
    except ImportError as e:
        logger.error(f"Error importing {class_names} from {module_name}: {e}")

# Network-related imports
safe_import("pynattas.classes.generic_lightning_module", [
    "GenericLightningNetwork",
    "GenericLightningNetwork_Custom",
    "GenericOD_YOLOv3",
    "GenericOD_YOLOv3_SmallObjects"
])

safe_import("pynattas.classes.generic_network", ["GenericNetwork"])

# Early stopping imports
safe_import("pynattas.classes.my_early_stopping", ["EarlyStopping", "TrainEarlyStopping"])