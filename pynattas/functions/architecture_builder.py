import random
import configparser
import logging
from typing import List, Dict, Optional
from pynattas import vocabulary

# Set up logging configuration
logger = logging.getLogger(__name__)

# Constants
ARCHITECTURE_ENDER = "|"


def generate_random_architecture_code(max_layers: int, config_path: Optional[str] = 'config.ini') -> str:
    """
    Generates a random architecture code with a variable number of layers, based on a specified task.

    Parameters:
    max_layers (int): Maximum number of layers for the architecture.
    config_path (Optional[str]): Path to the configuration file (defaults to 'config.ini').

    Returns:
    str: Generated architecture code.
    """
    task = get_task_from_config(config_path)

    min_layers = 3 if task == 'D' else 1
    encoder_layer_count = random.randint(min_layers, max_layers)
    
    # Generate architecture code
    architecture_code = ''.join(
        f"{generate_layer_code()}{ARCHITECTURE_ENDER}{generate_pooling_layer_code()}{ARCHITECTURE_ENDER}"
        for _ in range(encoder_layer_count)
    )
    
    logger.info(f"This architecture has {encoder_layer_count} encoder layers.")
    
    # Add head code and enders
    architecture_code += f"{generate_head_code(task, encoder_layer_count)}{ARCHITECTURE_ENDER}"
    architecture_code += ARCHITECTURE_ENDER

    return architecture_code


def get_task_from_config(config_path: str = 'config.ini') -> str:
    """
    Reads the task type from the configuration file.

    Parameters:
    config_path (str): Path to the configuration file.

    Returns:
    str: Task type ('C' or 'D', or 'S'). [TODO: Add 'S' support]

    Raises:
    ValueError: If the task is not found or invalid.
    """
    config = configparser.ConfigParser()
    try:
        config.read(config_path)
        task = config['Mode']['task']
        if task not in ['C', 'D', 'S']:
            raise ValueError(f"Invalid task '{task}' in configuration.")
        return task
    except (configparser.Error, KeyError) as e:
        logger.error(f"Error reading task from config: {e}")
        raise ValueError("Invalid configuration file or missing task information")


def generate_layer_code() -> str:
    """
    Generates a random code for a convolutional layer based on the configuration.

    Returns:
    str: Generated layer code.
    """
    layer_type = random.choice(list(vocabulary.convolution_layer_vocabulary.keys()))
    parameters = vocabulary.layer_parameters[vocabulary.convolution_layer_vocabulary[layer_type]]
    layer_code = f"L{layer_type}"

    config = configparser.ConfigParser()
    config.read('config.ini')
    section = vocabulary.convolution_layer_vocabulary[layer_type]

    for param in parameters:
        if param == 'activation':
            activation_code = random.choice(list(vocabulary.activation_functions_vocabulary.keys()))
            layer_code += f"a{activation_code}"
        elif param == 'num_blocks':
            continue  # Skip num_blocks for now (always set to 1)
        else:
            min_val = config.getint(section, f'min_{param}')
            max_val = config.getint(section, f'max_{param}')
            value = random.randint(min_val, max_val)
            layer_code += f"{vocabulary.parameter_vocabulary[param]}{value}"

    # Default block count is 1 for now
    layer_code += "n1"

    return layer_code


def generate_pooling_layer_code() -> str:
    """
    Generates a random code for a pooling layer based on the configuration.

    Returns:
    str: Generated pooling layer code.
    """
    pooling_type = random.choice(list(vocabulary.pooling_layer_vocabulary.keys()))
    return f"P{pooling_type}"


def generate_head_code(task: str, max_layers: int) -> str:
    """
    Generates the head code for the architecture based on the task.

    Parameters:
    task (str): The task type ('C' or 'D').
    max_layers (int): Number of encoder layers.

    Returns:
    str: Generated head code.

    Raises:
    ValueError: If the task is invalid.
    """
    def add_outchannel_indices_to_head(head_type: str, max_layers: int) -> str:
        """
        Adds specific outchannel indices to the head code based on the task type 'D'.

        Parameters:
        head_type (str): The initial head type code.
        max_layers (int): Number of encoder layers.

        Returns:
        str: Updated head code with outchannel indices.
        """
        if head_type in ['Y', 'S']:
            outchannels_indexes = generate_even_numbers((max_layers * 2) - 1)
            if head_type == 'Y':
                head_type += f'u{outchannels_indexes[0]}v{outchannels_indexes[1]}w{outchannels_indexes[2]}'
            elif head_type == 'S':
                head_type += f'u{outchannels_indexes[1]}v{outchannels_indexes[2]}'
        return head_type
    
    if task == 'C':
        head_type = random.choice(list(vocabulary.head_vocabulary_C.keys()))
    elif task == 'D':
        head_type = random.choice(list(vocabulary.head_vocabulary_D.keys()))
        head_type = add_outchannel_indices_to_head(head_type, max_layers)
    else:
        logger.error(f"Invalid task '{task}' for head code generation.")
        raise ValueError(f"Task {task} is not supported for head code generation.")
    
    return f"H{head_type}"


def generate_even_numbers(max_value: int) -> List[int]:
    """
    Generates a sorted list of three distinct even numbers within the range [0, max_value].

    Parameters:
    max_value (int): The upper limit for generating even numbers.

    Returns:
    List[int]: A list of three distinct even numbers.

    Raises:
    ValueError: If the max_value is less than 4 or there are not enough even numbers.
    """
    if max_value < 4:
        raise ValueError("max_value must be at least 4 to generate three even numbers.")
    
    even_numbers = [i for i in range(0, max_value + 1) if i % 2 == 0]
    if len(even_numbers) < 3:
        raise ValueError("Not enough even numbers to choose from.")
    
    selected_numbers = random.sample(even_numbers[:-1], 2) + [even_numbers[-1]]
    return sorted(selected_numbers)


def parse_architecture_code(architecture_code: str) -> List[Dict[str, str]]:
    """
    Parses the architecture code into a list of dictionaries representing the layers and their parameters.

    Parameters:
    architecture_code (str): The architecture code to be parsed.

    Returns:
    List[Dict[str, str]]: A list of parsed layers with their parameters.
    """
    segments = architecture_code.split(ARCHITECTURE_ENDER)[:-1] # Remove the last empty segment
    parsed_layers = []

    for segment in segments:
        if not segment:
            logger.warning("Empty segment found in architecture code.")
            continue  # Skip empty segments
        
        segment_info = parse_segment(segment)
        parsed_layers.append(segment_info)

    return parsed_layers


def parse_segment(segment: str) -> Dict[str, str]:
    """
    Parses an individual segment of the architecture code.

    Parameters:
    segment (str): The architecture segment.

    Returns:
    Dict[str, str]: Parsed segment details as a dictionary.
    """
    segment_type_code = segment[0]
    layer_type_code = segment[1]
    layer_type = get_layer_type(segment_type_code, layer_type_code)

    segment_info = {'layer_type': layer_type}
    params = segment[2:]  # Parameters portion of the segment

    i = 0
    while i < len(params):
        param_code = params[i]
        i += 1
        param_value = ""

        # Collect digits for parameter value
        while i < len(params) and params[i].isdigit():
            param_value += params[i]
            i += 1

        if param_code in vocabulary.parameter_vocabulary:
            param_name = vocabulary.parameter_vocabulary[param_code]
            segment_info[param_name] = int(param_value) if param_value.isdigit() else param_value

    return segment_info


def get_layer_type(segment_type_code: str, layer_type_code: str) -> str:
    """
    Returns the layer type based on segment type and layer type code.

    Parameters:
    segment_type_code (str): The segment type code ('L', 'P', 'H').
    layer_type_code (str): The layer type code.

    Returns:
    str: Layer type.
    """
    if segment_type_code == 'L':
        return vocabulary.convolution_layer_vocabulary[layer_type_code]
    elif segment_type_code == 'P':
        return vocabulary.pooling_layer_vocabulary[layer_type_code]
    elif segment_type_code == 'H':
        return vocabulary.head_vocabulary_C.get(layer_type_code, vocabulary.head_vocabulary_D.get(layer_type_code, 'Unknown'))
    else:
        raise ValueError(f"Unknown segment type code: {segment_type_code}")





if __name__ == '__main__':
    print(generate_random_architecture_code(5, 'config.ini'))