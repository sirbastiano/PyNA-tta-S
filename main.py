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
import pynattas as pnas
from datetime import datetime
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
        print("\n*** Network Architecture Search ***\n")

        # Define NAS parameters and run GA
        max_layers = config.getint('NAS', 'max_layers')
        max_iterations = int(config['GA']['max_iterations'])
        population_size = int(config['GA']['population_size'])
        log_path = str(config['GA']['logs_dir_GA'])
        mating_pool_cutoff = float(config['GA']['mating_pool_cutoff'])
        mutation_probability = float(config['GA']['mutation_probability'])
        nas_result = pnas.optimizers.ga.ga_optimizer(
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

    # Use NAS result if available, otherwise load from config
    architecture_code = nas_result['position'] if nas_check else config['NAS']['architecture_code']
    parsed_layers = pnas.functions.architecture_builder.parse_architecture_code(architecture_code)

    # HT
    if ht_check:
        print("\n*** Hyperparameter Tuning ***\n")
        print('HT Layers:', parsed_layers)

        # Define HT search space
        search_space = pnas.functions.utils.generate_layers_search_space(parsed_layers)
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
            ht_result = pnas.optimizers.gwo.gwo_optimizer(
                parsed_layers=parsed_layers,
                architecture_code=architecture_code,
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
            ht_result = pnas.optimizers.pso.pso_optimizer(
                parsed_layers=parsed_layers,
                architecture_code=architecture_code,
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
        pnas.functions.utils.save_and_print_results(
            ht_result,
            search_space,
            architecture_code,
            parsed_layers,
            optimizer_selection,
            max_iterations,
            log_path,
        )

        best_position = ht_result['position']
        best_parsed_layers = ht_result['parsed_layers']
        best_architecture_code = pnas.functions.architecture_builder.generate_code_from_parsed_architecture(best_parsed_layers)

        # Print and write HT results
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"HT completed. Final best architecture starting from {architecture_code} is {best_architecture_code}.")
        with open(f"{log_path}/ht_{current_datetime}_results.txt", "w") as file:
            file.write(f"Final best architecture starting from {architecture_code} is {best_architecture_code}.\nFull results:\n{ht_result}")

        architecture_code = best_architecture_code
        parsed_layers = best_parsed_layers

    end_time = time.time()
    print(f"Process finished in {end_time - start_time} s. Starting final run...")

    pnas.functions.fitness.compute_fitness_value(parsed_layers=parsed_layers, is_final=True)
