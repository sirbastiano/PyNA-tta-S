import random
import configparser
from .. import configuration


def generate_random_architecture_code(max_layers):
    # Get task
    config = configparser.ConfigParser()
    config.read('config.ini')
    task = config['Mode']['task']

    #architecture_code = "B"
    architecture_code = ""

    min_layers = 1
    if task == 'D':
        min_layers = 3

    encoder_layer_count = 0
    for _ in range(random.randint(min_layers, max_layers)):
        encoder_layer_count += 1
        architecture_code += generate_layer_code()
        architecture_code += "E"
        architecture_code += generate_pooling_layer_code()
        architecture_code += "E"
    
    print(f"This architecture has {encoder_layer_count} encoder layers.")
    architecture_code += generate_head_code(task, encoder_layer_count)
    architecture_code += "E"

    # Insert ender
    architecture_code += "E"

    return architecture_code


def generate_layer_code():
    layer_type = random.choice(list(configuration.convolution_layer_vocabulary.keys()))
    parameters = configuration.layer_parameters[configuration.convolution_layer_vocabulary[layer_type]]
    layer_code = f"L{layer_type}"

    config = configparser.ConfigParser()
    config.read('config.ini')
    section = configuration.convolution_layer_vocabulary[layer_type]

    for param in parameters:
        if param == 'activation':
            # Randomly select an activation function
            activation_code = random.choice(list(configuration.activation_functions_vocabulary.keys()))
            layer_code += f"a{activation_code}"
        elif param == 'num_blocks':
            # For now, n is always 1
            continue
        else:
            # Correctly fetch min and max values using getint
            min_val = config.getint(section, 'min_' + param)
            max_val = config.getint(section, 'max_' + param)
            value = random.randint(min_val, max_val)

            code = configuration.parameter_vocabulary[param]
            layer_code += f"{code}{str(value)}"

    # Insert number of same blocks to concatenate before a pooling layer. Currently 1. For future developments.
    layer_code += "n1"

    return layer_code


def generate_pooling_layer_code():
    pooling_type = random.choice(list(configuration.pooling_layer_vocabulary.keys()))
    return f"P{pooling_type}"


def generate_head_code(task, max_layers):
    if task not in ['C', 'D']:
        print(f'Error gathering task when generating architecture code. {task} is not a valid task.')
        exit()

    if task == 'C':
        head_type = random.choice(list(configuration.head_vocabulary_C.keys()))
    elif task == 'D':
        head_type = random.choice(list(configuration.head_vocabulary_D.keys()))
        #parameters = configuration.layer_parameters[configuration.convolution_layer_vocabulary[head_type]]
        #for param in parameters:
        if head_type == 'Y':
            outchannels_indexes = generate_even_numbers((max_layers*2)-1)
            head_type += f'u{outchannels_indexes[0]}'
            head_type += f'v{outchannels_indexes[1]}'
            head_type += f'w{outchannels_indexes[2]}'
    #elif task == 'S':
    #    print('Error gathering task when generating architecture code. Segmentation task is not yet implemented.')
    #    exit()
    
    return f"H{head_type}"


def generate_even_numbers(max_value):
    if max_value < 4:
        raise ValueError("The maximum value must be at least 4 to get three different even numbers.")

    # Generate all possible even numbers within the range
    even_numbers = [i for i in range(0, max_value + 1) if i % 2 == 0]
    
    if len(even_numbers) < 3:
        raise ValueError("Not enough even numbers in the range to choose from.")

    # Ensure the largest even number is included
    max_even = even_numbers[-1]
    
    # Select 2 different even numbers from the list excluding the maximum even number
    selected_numbers = random.sample(even_numbers[:-1], 2)
    
    # Add the largest even number to the list
    selected_numbers.append(max_even)
    
    # Sort the numbers in increasing order
    selected_numbers.sort()
    
    return selected_numbers


def parse_architecture_code(architecture_code):
    # Get task
    config = configparser.ConfigParser()
    config.read('config.ini')
    task = config['Mode']['task']
    
    segments = architecture_code.split('E')[:-1]
    parsed_layers = []

    for segment in segments:
        if not segment:  # Skip empty segments
            continue
        
        segment_type_code = segment[0]
        layer_type_code = segment[1]
        
        # Determine the segment's layer type and corresponding parameters
        if segment_type_code == 'L':
            layer_type = configuration.convolution_layer_vocabulary.get(layer_type_code, "Unknown")
            param_definitions = configuration.layer_parameters.get(layer_type, [])
        elif segment_type_code == 'P':
            layer_type = configuration.pooling_layer_vocabulary.get(layer_type_code, "Unknown")
            param_definitions = configuration.layer_parameters.get(layer_type, [])
        elif segment_type_code == 'H':
            if task == 'C':
                layer_type = configuration.head_vocabulary_C.get(layer_type_code, "Unknown")
            elif task == 'D':
                layer_type = configuration.head_vocabulary_D.get(layer_type_code, "Unknown")
            else:
                print(f'Error gathering task when generating architecture code. {task} is not a valid task.')
                exit()
            param_definitions = configuration.layer_parameters.get(layer_type, [])
        
        # Initialize the dictionary for this segment with its type
        segment_info = {'layer_type': layer_type}
        
        # Process remaining characters based on the expected parameters for this type
        params = segment[2:]  # All after layer type code
        i = 0
        
        while i < len(params):
            param_code = params[i]
            i += 1
            param_value_code = ""
            
            # Collect all consecutive digits for parameter value
            while i < len(params) and params[i].isdigit():
                param_value_code += params[i]
                i += 1
            
            if not param_value_code:
                if i < len(params):
                    param_value_code = params[i]
                    i += 1
            
            # Find the parameter name from the code
            for param_name, code in configuration.parameter_vocabulary.items():
                if code == param_code:
                    # Add parameter to segment info, converting numeric values
                    if param_value_code.isdigit():
                        segment_info[param_name] = int(param_value_code)
                    else:
                        # Map activation codes to their respective names
                        if param_name == 'activation':
                            segment_info[param_name] = configuration.activation_functions_vocabulary.get(param_value_code, "Unknown")
                        else:
                            segment_info[param_name] = param_value_code
                    break
        
        parsed_layers.append(segment_info)

    return parsed_layers



def generate_code_from_parsed_architecture(parsed_layers):
    architecture_code = ""

    # Unify Head vocabularies
    head_vocabulary = configuration.head_vocabulary_C | configuration.head_vocabulary_D
    
    # Utilize the provided configuration directly
    reverse_convolution_layer_vocabulary = {v: k for k, v in configuration.convolution_layer_vocabulary.items()}
    reverse_pooling_layer_vocabulary = {v: k for k, v in configuration.pooling_layer_vocabulary.items()}
    reverse_head_vocabulary = {v: k for k, v in head_vocabulary.items()}
    reverse_activation_functions_vocabulary = {v: k for k, v in configuration.activation_functions_vocabulary.items()}

    for layer in parsed_layers:
        layer_type = layer['layer_type']
        segment_code = ""
        
        # Prepend the type code with "L", "P", or "H" based on the layer type
        if layer_type in reverse_convolution_layer_vocabulary:
            segment_code += "L" + reverse_convolution_layer_vocabulary[layer_type]
        elif layer_type in reverse_pooling_layer_vocabulary:
            segment_code += "P" + reverse_pooling_layer_vocabulary[layer_type]
        elif layer_type in reverse_head_vocabulary:
            segment_code += "H" + reverse_head_vocabulary[layer_type]

        # Append each parameter and its value
        for param_name, param_value in layer.items():
            if param_name == 'layer_type':  # Skip 'layer_type' as it's already processed
                continue
            
            if param_name in configuration.parameter_vocabulary:
                param_code = configuration.parameter_vocabulary[param_name]
                
                # Special handling for activation parameters
                if param_name == 'activation':
                    param_value = reverse_activation_functions_vocabulary.get(param_value, param_value)
                
                segment_code += param_code + str(param_value)
        
        # Finalize the segment and add it to the architecture code
        architecture_code += segment_code + "E"
    
    # Ensure the architecture code properly ends with "EE"
    return architecture_code + "E"
