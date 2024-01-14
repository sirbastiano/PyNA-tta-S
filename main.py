"""
This script orchestrates Network Architecture Search (NAS) and Hyperparameter Tuning (HT) using optimization algorithms.
It uses 'config.ini' to determine operation modes and optimization parameters.

Modes:
1. NAS: Finds optimal neural network architectures.
2. HT: Fine-tunes hyperparameters for a specified or NAS-derived architecture.

It supports Genetic Algorithm (GA) for NAS and Grey Wolf Optimizer (GWO) or Particle Swarm Optimizer (PSO) for HT.

Flow:
- Load configurations from 'config.ini'.
- Perform NAS if nas_check is True.
- Perform HT if ht_check is True, using either NAS-derived or specified architecture.
- Save and print results.
- Exits with an error if neither mode is selected.

External dependencies include configparser, GA for NAS, and GWO/PSO for HT.
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
    if not nas_check and not ht_check:
        print("Error: No mode selected. Check config_optimizer.ini for verification.")
        exit()

    nas_result = None
    ht_result = None

    # NAS
    if nas_check:
        # Define NAS parameters and run GA
        max_layers = config.getint('NAS', 'max_layers')
        max_iterations = int(config['GA']['max_iterations'])
        population_size = int(config['GA']['population_size'])
        log_path = str(config['GA']['logs_dir_GA'])
        mating_pool_cutoff = float(config['GA']['mating_pool_cutoff'])
        mutation_probability = float(config['GA']['mutation_probability'])
        nas_result = ga.ga_optimizer(
            max_layers=max_layers,
            max_iter=max_iterations,
            n_individuals=population_size,
            mating_pool_cutoff=mating_pool_cutoff,
            mutation_probability=mutation_probability,
            logs_directory=log_path,
        )
        # Save NAS results
        #functions.utils.save_nas_results

        # Print and write NAS results
        print(f"NAS completed. Best architecture: {nas_result['position']}")
        with open(f"{log_path}/nas_results.txt", "w") as file:
            file.write(f"Best architecture: {nas_result['position']}")

    # HT
    if ht_check:
        # Use NAS result if available, otherwise load from config
        architecture_code = nas_result['position'] if nas_check else config['NAS']['architecture_code']
        layers = functions.utils.parse_architecture_code(architecture_code)
        print('HT Layers:', layers)

        # Define HT search space
        search_space = {}
        if nas_check:
            search_space_layers = functions.utils.generate_layers_search_space(layers)
            for parameter in search_space_layers.keys():
                search_space[parameter] = search_space_layers.get(parameter)

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
            max_iterations = int(config['GWO']['max_iterations'])
            population_size = int(config['GWO']['population_size'])
            log_path = str(config['GWO']['logs_dir_GWO'])
            ht_result = gwo.gwo_optimizer(
                architecture=layers,
                search_space=search_space,
                max_iter=max_iterations,
                n_wolves=population_size,
                logs_directory=log_path,
            )
        elif optimizer_selection == '2':
            max_iterations = int(config['PSO']['max_iterations'])
            population_size = int(config['PSO']['population_size'])
            log_path = str(config['PSO']['logs_dir_PSO'])
            cognitive_coefficient = float(config['PSO']['cognitive_coefficient'])
            social_coefficient = float(config['PSO']['social_coefficient'])
            inertia_coefficient = float(config['PSO']['inertia_coefficient'])
            ht_result = pso.pso_optimizer(
                architecture=layers,
                search_space=search_space,
                max_iter=max_iterations,
                n_particles=population_size,
                c1=cognitive_coefficient,
                c2=social_coefficient,
                w=inertia_coefficient,
                logs_directory=log_path,
            )
        else:
            print('Error in HT optimizer selection. Check config_optimizer.ini for verification.')
            exit()

        # Save HT results
        functions.utils.save_and_print_results(
            ht_result,
            search_space,
            architecture_code,
            layers,
            optimizer_selection,
            max_iterations,
            log_path,
        )

        # Print and write HT results
        print(f"HT completed. Best hyperparameters for architecture {architecture_code}: {ht_result}")
        with open(f"{log_path}/ht_{architecture_code}_results.txt", "w") as file:
            file.write(f"Best hyperparameters for architecture {architecture_code}: {ht_result}")

    end_time = time.time()
    print(f"Process finished in {end_time - start_time} s.")
