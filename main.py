"""
This script orchestrates the optimization process for both Network Architecture Search (NAS) and Hyperparameter Tuning
(HT) using various optimization algorithms. It reads configurations from a 'config.ini' file to determine the modes of
operation and parameters for the optimization processes.

The script supports two primary modes:
1. Network Architecture Search (NAS): Optimizes the architecture of a neural network based on the specified architecture
code.
2. Hyperparameter Tuning (HT): Fine-tunes hyperparameters like learning rate and batch size.

The script allows the use of different optimization algorithms, including Grey Wolf Optimizer (GWO), Particle Swarm
Optimizer (PSO), and Genetic Algorithm (GA), based on the user's selection in the configuration file.

Flow:
- Load configuration from 'config.ini'.
- Check and validate the selected modes (NAS, HT).
- For NAS mode:
  - Parse the architecture code and generate a search space specific to NAS.
- For HT mode:
  - Define the search space for hyperparameters like learning rate and batch size.
- Execute the chosen optimization algorithm based on the optimizer selection in the config.
- Save and print the results of the optimization process.

The script requires the following external dependencies:
- configparser: For parsing the 'config.ini' file.
- gwo, pso, ga: Optimization algorithm modules for GWO, PSO, and GA respectively.
- functions: A module containing utility functions like parsing architecture codes and generating search spaces.

The script is designed to be versatile, allowing users to easily switch between different modes and optimization
algorithms through configuration changes, without needing to alter the code.


NOTES:
12/01/2024
Current implementation works as follows:
* if nas_check and ht_check are both True, nas and ht search are done at the same time.
* if only ht_check is selected, it will do tuning on a model with the selected architecture but default parameters.
  Change the default parameters for the layers of interest to change it. This means, however, that this default ht
  model is less flexible, as the parameters will be the same for every layer of the same type.
"""

# Imports
import configparser
from optimizers import gwo, pso, ga
import functions
import time

if __name__ == '__main__':
    start_time = time.time()

    # Load configuration from config.ini
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Check modes
    nas_check = config.getboolean(section='Mode', option='network_architecture_search')
    ht_check = config.getboolean(section='Mode', option='hyperparameter_tuning')
    if nas_check is False and ht_check is False:
        print("Error selecting mode. Check config_optimizer.ini for verification.")
        exit()

    # Selected architecture
    architecture_code = str(config['NAS']['architecture_code'])
    layers = functions.utils.parse_architecture_code(architecture_code)
    print('Layers:', layers)

    # Search space
    search_space = {}
    if nas_check:
        search_space_nas = functions.utils.generate_nas_search_space(layers)
        for parameter in search_space_nas.keys():
            search_space[parameter] = search_space_nas.get(parameter)

    if ht_check:
        search_space['log_learning_rate'] = (
            float(config['Search Space']['log_lr_min']),
            float(config['Search Space']['log_lr_max']),
        )
        search_space['batch_size'] = (
            float(config['Search Space']['bs_min']),
            float(config['Search Space']['bs_max']),
        )
    print('Search space:', search_space)

    # Retrieve optimizer information
    optimizer_selection = config['Optimizer']['optimizer_selection']
    if optimizer_selection == '1':
        # Grey Wolf Optimizer
        max_iterations = int(config['GWO']['max_iterations'])
        population_size = int(config['GWO']['population_size'])
        log_path = str(config['GWO']['logs_dir_GWO'])
        best_fit = gwo.gwo_optimizer(
            architecture=layers,
            search_space=search_space,
            max_iter=max_iterations,
            n_wolves=population_size,
            logs_directory=log_path,
        )

    elif optimizer_selection == '2':
        # Particle Swarm Optimizer
        max_iterations = int(config['PSO']['max_iterations'])
        population_size = int(config['PSO']['population_size'])
        log_path = str(config['PSO']['logs_dir_PSO'])
        cognitive_coefficient = float(config['PSO']['cognitive_coefficient'])
        social_coefficient = float(config['PSO']['social_coefficient'])
        inertia_coefficient = float(config['PSO']['inertia_coefficient'])
        best_fit = pso.pso_optimizer(
            architecture=layers,
            search_space=search_space,
            max_iter=max_iterations,
            n_particles=population_size,
            c1=cognitive_coefficient,
            c2=social_coefficient,
            w=inertia_coefficient,
            logs_directory=log_path,
        )

    elif optimizer_selection == '3':
        # Genetic Algorithm with single-point crossover
        max_iterations = int(config['GA']['max_iterations'])
        population_size = int(config['GA']['population_size'])
        log_path = str(config['GA']['logs_dir_GA'])
        mating_pool_cutoff = float(config['GA']['mating_pool_cutoff'])
        crossover_cutoff = int(config['GA']['crossover_cutoff'])
        mutation_probability = float(config['GA']['mutation_probability'])
        best_fit = ga.ga_optimizer(
            architecture=layers,
            search_space=search_space,
            max_iter=max_iterations,
            n_individuals=population_size,
            mating_pool_cutoff=mating_pool_cutoff,
            crossover_cutoff=crossover_cutoff,
            mutation_probability=mutation_probability,
            logs_directory=log_path,
        )

    else:
        print('Error in optimizer selection. Check config_optimizer.ini for verification.')
        exit()

    # Final saves and prints
    functions.utils.save_and_print_results(
        best_fit=best_fit,
        search_space=search_space,
        architecture_code=architecture_code,
        layers=layers,
        optimizer_selection=optimizer_selection,
        max_iterations=max_iterations,
        log_path=log_path
    )

    end_time = time.time()
    print(f"Process finished in {end_time-start_time} s.")
