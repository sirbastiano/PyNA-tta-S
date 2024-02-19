import random
import configparser
from .. import configuration


def generate_random_architecture_code(max_layers):
    #architecture_code = "B"
    architecture_code = ""

    for _ in range(random.randint(1, max_layers)):
        architecture_code += generate_layer_code()
        architecture_code += "E"
        architecture_code += generate_pooling_layer_code()
        architecture_code += "E"
    
    architecture_code += generate_head_code()
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
    pooling_code = f"P{pooling_type}"
    return pooling_code


def generate_head_code():
    head_type = random.choice(list(configuration.head_vocabulary.keys()))
    head_code = "HC"
    return head_code


def parse_architecture_code(architecture_code):
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
            layer_type = configuration.head_vocabulary.get(layer_type_code, "Unknown")
            param_definitions = configuration.layer_parameters.get(layer_type, [])
        
        # Initialize the dictionary for this segment with its type
        segment_info = {'layer_type': layer_type}
        
        # Process remaining characters based on the expected parameters for this type
        params = segment[2:]  # All after layer type code
        
        for i in range(0, len(params), 2):  # Process in pairs
            if i + 1 < len(params):
                param_code = params[i]
                param_value_code = params[i + 1]
                
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
    
    # Utilize the provided configuration directly
    reverse_convolution_layer_vocabulary = {v: k for k, v in configuration.convolution_layer_vocabulary.items()}
    reverse_pooling_layer_vocabulary = {v: k for k, v in configuration.pooling_layer_vocabulary.items()}
    reverse_head_vocabulary = {v: k for k, v in configuration.head_vocabulary.items()}
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
