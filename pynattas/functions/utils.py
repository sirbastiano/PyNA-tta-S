import configparser
import os
from .. import configuration


# TODO: add functions for plotting the population at each iteration. Maybe add function to create the gif as well.

def generate_layers_search_space(parsed_layers):
    config = configparser.ConfigParser()
    config.read('config.ini')

    search_space = {}
    for index, layer in enumerate(parsed_layers):
        layer_type = layer['layer_type']

        # Get the parameters for the specific layer type
        if layer_type in configuration.layer_parameters:
            for param in configuration.layer_parameters[layer_type]:
                if param != 'activation' and param != 'num_blocks':
                    range_key = f"min_{param}", f"max_{param}"
                    param_key = f"{layer_type}_{index}_{param}"  # Unique key using layer type, index, and parameter name
                    search_space[param_key] = (
                        float(config[layer_type][range_key[0]]),
                        float(config[layer_type][range_key[1]]),
                    )

    return search_space


def generate_bs_lr_search_space(lr_check = True, bs_check = True):
    config = configparser.ConfigParser()
    config.read('config.ini')
    search_space = {}

    if lr_check:
        log_lr_min = float(config['Search Space']['log_lr_min'])
        log_lr_max = float(config['Search Space']['log_lr_max'])
        search_space['log_learning_rate'] = (log_lr_min, log_lr_max)
    
    if bs_check:
        bs_min = float(config['Search Space']['bs_min'])
        bs_max = float(config['Search Space']['bs_max'])
        search_space['batch_size'] = (bs_min, bs_max)

    return search_space


def update_parsed_layers_and_extract_specials(parsed_layers, position, search_space_keys):
    """
    Updates parsed_layers with values from position based on keys in search_space_keys.
    
    Parameters:
    - parsed_layers (list): A list of dictionaries, where each dictionary represents layer parameters.
    - position (list): A list of values corresponding to the keys in search_space_keys.
    - search_space_keys (list): A list of keys indicating where to update values in parsed_layers.
    """
    temp_position = position.copy()

    # Pop the values for log_learning_rate and batch_size
    batch_size = temp_position.pop() if 'batch_size' in search_space_keys[-2:] else None
    log_learning_rate = temp_position.pop() if 'log_learning_rate' in search_space_keys[-2:] else None
    
    # Remove them from search_space_keys if present
    search_space_keys = [key for key in search_space_keys if key not in ['log_learning_rate', 'batch_size']]

    for index, key in enumerate(search_space_keys):
        # Parse the key structure to identify layer index and parameter
        parts = key.split('_')
        layer_index = int(parts[1])
        parameter = '_'.join(parts[2:])  # Handle cases where parameter names might contain '_'
        
        # Update the corresponding dictionary in parsed_layers
        if layer_index < len(parsed_layers) and parameter in parsed_layers[layer_index]:
            parsed_layers[layer_index][parameter] = round(temp_position[index])
    
    return parsed_layers, log_learning_rate, batch_size


def show_population(population, generation, logs_dir, historical_best_fitness, fittest_individual):
    txt_filename = f'GA_generation_{generation}.txt'
    txt_filepath = os.path.join(logs_dir, txt_filename)
    with open(txt_filepath, 'w') as txt_file:
        for j, ind in enumerate(population):
            txt_file.write(f"Individual {j} - Genes: {ind.chromosome}, Fitness: {ind.fitness}\n")
        txt_file.write(
            f"Historical Fittest individual - Genes: {fittest_individual}, Fitness: {historical_best_fitness}")

    print(f"Text file saved: {txt_filepath}")


def show_particle_swarm(swarm, search_space, global_best_position, global_best_fitness, logs_directory, iteration):
    t = iteration

    with open(os.path.join(logs_directory, f'iteration_{t}.txt'), 'w') as file:
        file.write("Particle Information:\n")
        for i, particle in enumerate(swarm):
            file.write(
                f"Particle {i} - Position: {particle.current_position}, "
                f"Velocity: {particle.current_velocity}, "
                f"Fitness: {particle.current_fitness}\n"
            )
        file.write(f"Global Best - Position: {global_best_position}, Fitness: {global_best_fitness}\n")


def show_wolf_population(population, search_space, iteration, logs_dir):
    txt_filename = f'iteration_{iteration}.txt'
    txt_filepath = os.path.join(logs_dir, txt_filename)
    with open(txt_filepath, 'w') as txt_file:
        for i, wolf in enumerate(population):
            txt_file.write(f"Wolf {i} - Position: {wolf.position}, Fitness: {wolf.fitness}\n")

    print(f"Text file saved: {txt_filepath}")


def calculate_statistics(data_list, attribute):
    attribute_values = [getattr(obj, attribute) for obj in data_list]
    mean = sum(attribute_values) / len(attribute_values)
    sorted_values = sorted(attribute_values)
    n = len(sorted_values)
    if n % 2 == 1:
        median = sorted_values[n // 2]
    else:
        middle_left = sorted_values[n // 2 - 1]
        middle_right = sorted_values[n // 2]
        median = (middle_left + middle_right) / 2
    return mean, median


def save_and_print_results(best_fit, search_space, architecture_code, layers, optimizer_selection, max_iterations, log_path):
    print(f"\nRUN COMPLETED! The best fit is: {best_fit}")
    search_space_keys = list(search_space.keys())
    best_fit_position = best_fit['position']

    # Final NAS print
    for i in range(len(search_space_keys)):
        print(f"{search_space_keys[i]}: {round(float(best_fit_position[i]))}")

    # File saved
    txt_filename = f'{architecture_code}_iteration_optim_{optimizer_selection}_iteration_{max_iterations}.txt'
    txt_filepath = os.path.join(log_path, txt_filename)
    with open(txt_filepath, 'w') as txt_file:
        txt_file.write(f"For the following architecture: {architecture_code}\n")
        txt_file.write(f"{layers}\n")
        txt_file.write(f"Best fit: {best_fit}\n")
        for i in range(len(search_space_keys)):
            if search_space_keys[i] == 'log_learning_rate':
                txt_file.write(f"{search_space_keys[i]}: {float(best_fit_position[i])}\n")
            else:
                txt_file.write(f"{search_space_keys[i]}: {round(float(best_fit_position[i]))}\n")
    print(f"Final text file saved: {txt_filepath}")

