import torch
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
#from pytorch.tuner import Tuner
import configparser
import os
import time
from datetime import datetime
from .. import classes
from datasets.L0_thraws_classifier.dataset_weighted import SentinelDataset, SentinelDataModule
from datasets.wake_classifier.dataset import xAIWakesDataModule


def compute_fitness_value(parsed_layers, log_learning_rate=None, batch_size=None, is_final=False):
    """
    Computes the fitness value for a given network s and hyperparameters in NAS.

    This function constructs and trains a PyTorch Lightning model based on the provided s and
        hyperparameters.
    It uses the xAIWakesDataModule for data loading and preprocessing. The fitness is calculated based on the test
        accuracy and F1 score obtained after training the model.

    Parameters:
    position (list): A list of hyperparameter values corresponding to the keys.
    keys (list): A list of hyperparameter names.
    s (list): A list of dictionaries defining the neural network architecture.

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
    log_lr = log_learning_rate if log_learning_rate != None else config.getfloat(section='Search Space', option='default_log_lr')
    lr = 10**log_lr
    bs = batch_size if batch_size != None else config.getint(section='Search Space', option='default_bs')

    # DATA
    csv_file = config['Dataset']['csv_path']
    root_dir = config['Dataset']['data_path']
    num_classes = config.getint(section='Dataset', option='num_classes')

    """
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
    #"""
    
    print(f"\n\n***\n\n{parsed_layers}\n***\n\n")
    #exit()

    # MODEL
    if is_final == False:
        model = classes.GenericLightningNetwork(
            parsed_layers=parsed_layers,
            input_channels=in_channels,
            #input_height=256,
            #input_width=256,
            num_classes=num_classes,
            learning_rate=lr,
        )
        trainer = pl.Trainer(
            accelerator=accelerator,
            min_epochs=1,
            max_epochs=50,
            fast_dev_run=False,
            check_val_every_n_epoch=51,
            callbacks=[classes.TrainEarlyStopping(monitor='train_loss', mode="min", patience=2)]
        )

        ## Tuner
        #tuner = Tuner(trainer)
        #tuner.scale_batch_size(model, mode="binsearch")
        #tuner.lr_find(model)

        # Training
        training_start_time = time.time()
        trainer.fit(model, dm)
        training_time = time.time() - training_start_time

        trainer.validate(model, dm)

        # Test
        test_start_time = time.time()
        results = trainer.test(model, dm)
        test_time = time.time() - test_start_time
        #inference_time = test_time/55

        '''
        # Inference
        inference_start_time = time.time()
        model = classes.GenericLightningNetwork(
            parsed_layers=parsed_layers,
            input_channels=in_channels,
            #input_height=256,
            #input_width=256,
            num_classes=num_classes,
            learning_rate=lr,
        )
        checkpoint = torch.load(rf"/media/warmachine/DBDISK/Andrea/DicDic/logs/tb_logs/checkpoints/OptimizedModel_2024-04-24_11-53-18/version_0/checkpoints/epoch=5-step=246.ckpt")
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()       
        inference_time = time.time() - inference_start_time
        '''

        acc = results[0].get('test_accuracy')
        f1 = results[0].get('test_f1_score')
        mcc = results[0].get('test_mcc') 
        # fitness = (1/6)*(1*mcc+2*f1+3*acc)*(1/(training_time/10))

        num_param = sum(p.numel() for p in model.parameters())
        fitness = (20*acc)-(num_param/1000000)

        print(f"Training time: {training_time}")
        print(f"Accuracy: {acc}")
        print(f"F1 score: {f1}")
        print(f"MCC: {mcc}")
        print(f"Fitness: {fitness}")
        print("********")

        return fitness
    
    else:
        print("\nFINAL RUN ON OPTIMIZED ARCHITECTURE")

        # MODEL
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        checkpoints_path = config.get(section="Logging", option="checkpoints_dir")
        logger = TensorBoardLogger(save_dir=checkpoints_path, name=f"OptimizedModel_{current_datetime}")
        #checkpoint_filepath = os.path.join(checkpoints_path, f"OptimizedModel_{current_datetime}_Checkpoint.ckpt")
        model = classes.GenericLightningNetwork(
            parsed_layers=parsed_layers,
            input_channels=in_channels,
            # input_height=256,
            # input_width=256,
            num_classes=num_classes,
            learning_rate=lr,
        )
        trainer = pl.Trainer(
            accelerator=accelerator,
            min_epochs=1,
            max_epochs=50,
            logger=logger,
            #enable_checkpointing=True,
            #default_root_dir=checkpoints_path,
            check_val_every_n_epoch=1,
            callbacks=[EarlyStopping(monitor='val_loss', mode="min", patience=3)]
        )

        # Training
        training_start_time = time.time()
        trainer.fit(model, dm)
        training_time = time.time() - training_start_time

        trainer.validate(model, dm)

        # Test
        test_start_time = time.time()
        results = trainer.test(model, dm)
        test_time = time.time() - test_start_time
        #inference_time = test_time/55

        acc = results[0].get('test_accuracy')
        f1 = results[0].get('test_f1_score')
        mcc = results[0].get('test_mcc')
        #fitness = (1/6)*(1*mcc+2*f1+3*acc)*(1/(training_time/10))

        num_param = sum(p.numel() for p in model.parameters())
        fitness = (20*acc)-(num_param/1000000)

        print("FINAL RUN COMPLETED:")
        print(f"Training time: {training_time}")
        print(f"Accuracy: {acc}")
        print(f"F1 score: {f1}")
        print(f"MCC: {mcc}")
        print(f"Fitness: {fitness}")
        print("********")

        txt_filename = f'Optimized_Architecture_Final_Run_{current_datetime}.txt'
        txt_filepath = os.path.join(checkpoints_path, txt_filename)
        with open(txt_filepath, 'w') as txt_file:
            txt_file.write(f"For the following s:\n{parsed_layers}\n")
            txt_file.write(f"\nTraining time: {training_time}")
            txt_file.write(f"\nAccuracy: {acc}")
            txt_file.write(f"\nF1 score: {f1}")
            txt_file.write(f"\nMCC: {mcc}")
            txt_file.write(f"\nFitness: {fitness}")
        print(f"\nFinal run text file saved: {txt_filepath}")
