import torch
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
import configparser
from classes.generic_lightning_module import GenericLightningNetwork
from wake_classifier.dataset import xAIWakesDataModule
from L0_thraws_classifier.dataset_weighted import SentinelDataset, SentinelDataModule


def compute_fitness_value_nas(position, keys, architecture):
    """
    Computes the fitness value for a given network architecture and hyperparameters in NAS.

    This function constructs and trains a PyTorch Lightning model based on the provided architecture and
        hyperparameters.
    It uses the xAIWakesDataModule for data loading and preprocessing. The fitness is calculated based on the test
        accuracy and F1 score obtained after training the model.

    Parameters:
    position (list): A list of hyperparameter values corresponding to the keys.
    keys (list): A list of hyperparameter names.
    architecture (list): A list of dictionaries defining the neural network architecture.

    Returns:
    float: The computed fitness value, a weighted average of accuracy and F1 score.

    The function reads configuration settings from 'config.ini', sets up the model, data module, and trainer,
    and then trains and evaluates the model. The fitness value is a combination of the model's accuracy and F1 score
    on the test set, with more weight given to accuracy.
    """
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Torch stuff
    seed = config.getint(section='Computation', option='seed')
    pl.seed_everything(seed=seed, workers=True)  # For reproducibility
    torch.set_float32_matmul_precision("medium")  # to make lightning happy
    num_workers = config.getint(section='Computation', option='num_workers')
    accelerator = config.get(section='Computation', option='accelerator')

    # Get model parameters
    model_parameters = {}
    log_lr = config.getfloat(section='Search Space', option='default_log_lr')
    bs = config.getint(section='Search Space', option='default_bs')
    for index, parameter in enumerate(keys):
        model_parameters[parameter] = position[index]
        if parameter == 'log_learning_rate':
            log_lr = model_parameters['log_learning_rate']
        elif parameter == 'batch_size':
            bs = model_parameters['batch_size']
    lr = 10**log_lr

    # DATA
    csv_file = config['Dataset']['csv_path']
    root_dir = config['Dataset']['data_path']
    num_classes = config.getint(section='Dataset', option='num_classes')

    composed_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        #transforms.Lambda(lambda x: x.view(-1, 256, 256))  # Reshaping to CxHxW
    ])
    dataset = SentinelDataset(
        root_dir=root_dir,
        transform=composed_transform,
    )
    image, label = dataset[0]  # Load one image from the dataset
    in_channels = image.shape[0]  # Obtain the number of in channels. Height and Width are 256 x 256 due to transform
    dm = SentinelDataModule(
        root_dir=root_dir,
        batch_size=round(float(bs)),
        num_workers=num_workers,
        transform=composed_transform,
    )
    """
    dm = xAIWakesDataModule(
        csv_file=csv_file,
        root_dir=root_dir,
        batch_size=round(float(bs)),
        num_workers=num_workers,
        transform=torchvision.transforms.ToTensor(),
    )
    in_channels = 4
    """

    # MODEL
    model = GenericLightningNetwork(
        parsed_layers=architecture,
        input_channels=in_channels,
        #input_height=256,
        #input_width=256,
        num_classes=num_classes,
        learning_rate=lr,
        model_parameters=model_parameters,
    )
    trainer = pl.Trainer(
        accelerator=accelerator,
        min_epochs=1,
        max_epochs=11,
        fast_dev_run=False,
        check_val_every_n_epoch=50,
    )
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    results = trainer.test(model, dm)
    #acc = results[0].get('test_accuracy')
    #f1 = results[0].get('test_f1_score')
    # fitness = (4 * acc + 1 * f1) / 5
    # fitness = acc
    mcc = results[0].get('test_mcc')
    fitness = mcc

    return fitness
