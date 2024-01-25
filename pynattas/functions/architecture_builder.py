import random


def generate_random_architecture_code(max_layers):
    """
    Generates a random architecture code based on predefined rules.

    Parameters:
    max_layers (int): The maximum number of convolutional layers allowed in the architecture.

    Returns:
    str: A randomly generated architecture code.

    Rules for architecture code generation:
    - The last letter is 'C' representing a Classification Head.
    - The second to last letter is a pooling layer ('a' for AvgPool or 'M' for MaxPool).
    - The rest are triplets of convolutional layers, where each triplet consists of:
      - 'c' for Conv2D or 'm' for MBConv,
    - A number '1' (as only single consecutive layers are used in this implementation),
    - A letter for the activation function ('r' for ReLU or 'g' for GELU).
    """

    # FOR REPRODUCIBILITY, SET SEED

    ## Sanity check on input
    #if max_layers < 1:
    #    print("Error: Max layers should be bigger than 0.")
    #    exit()

    # Define possible choices for each part of the triplet
    conv_layers = ['c', 'm']  # Conv2D or MBConv
    activation_functions = ['r', 'g']  # ReLU or GELU
    pooling_layers = ['a', 'M']  # AvgPool or MaxPool
    heads = ['C']  # ClassificationHead

    architecture_code = ""

    # Generate convolutional layers
    for _ in range(random.randint(1, max_layers)):  # Reserve space for pooling and head layers
        conv_layer = random.choice(conv_layers)
        activation_function = random.choice(activation_functions)
        # Append the triplet to the architecture code
        architecture_code += f"{conv_layer}1{activation_function}"

    # Add a pooling layer
    architecture_code += random.choice(pooling_layers)

    # Add the classification head
    architecture_code += random.choice(heads)

    return architecture_code


def random_triplet_gene():
    """
    Generates a random triplet gene for convolutional layers.
    Triplet format: [layer type (c or m)] + [number of layers (currently always1)] + [activation function (r or g)]
    """

    # Define the options for each part of the triplet
    layer_types = ['c', 'm'] # Conv2D or MBConv
    activation_functions = ['r', 'g'] # ReLU or GELU

    # Randomly select one option from each part
    layer_type = random.choice(layer_types)
    activation_function = random.choice(activation_functions)

    # Construct and return the triplet gene
    triplet_gene= layer_type + '1' + activation_function

    return triplet_gene


def random_pooling_gene():
    """
    Generates a random gene for the pooling layer.
    """

    # Define the options for pooling layers
    pooling_types = ['a', 'M'] # AvgPool or MaxPool

    # Randomly select one option
    pooling_gene = random.choice(pooling_types)
    return pooling_gene


def random_head_gene():
    """
    Generates a random gene for the head layer.
    Currently, there is only one type of head ('C' for ClassificationHead),
    but this function

    can be extended if more types are added in the future.
    """
    # Define the options for head layers
    head_types = ['C']  # Currently only ClassificationHead

    # Randomly select one option
    head_gene = random.choice(head_types)
    return head_gene
