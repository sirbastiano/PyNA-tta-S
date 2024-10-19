import torch
import random
from typing import List, Optional
from configparser import ConfigParser
import copy
from pynattas.utils import layerCoder
from pynattas.utils.layerCoder import ARCHITECTURE_ENDER as ENDER
from pynattas.builder.netBuilder import GenericNetwork



class Individual:
    def __init__(self, config_path: Optional[str] = 'config.ini', verbose: bool = False):
        """
        Initializes an individual with a random architecture code, converts it to a chromosome, 
        and sets the fitness to zero.
        
        Parameters:
        max_layers (int): Maximum number of layers for the architecture.
        """
        self.config_path = config_path        
        
        # Private attributes:
        self._fitness: float = 0.0
        self._architecture: str = self.generate_random_architecture_code()
        # Public attributes:
        self.chromosome: List[str] = self.architecture2chromosome(input_architecture=self.architecture)

    @property
    def architecture(self) -> str:
        return self._architecture

    def generate_random_architecture_code(self) -> str:
        """
        Generates a random architecture code with a variable number of layers, based on a specified task.

        Parameters:
        max_layers (int): Maximum number of layers for the architecture.
        config_path (Optional[str]): Path to the configuration file (defaults to 'config.ini').

        Returns:
        str: Generated architecture code.
        """
        
        task = layerCoder.get_task_from_config(self.config_path)

        min_layers = 3 if task == 'D' else self.config.getint('GeneralNetwork', 'min_layers')
        max_layers = self.config.getint('GeneralNetwork', 'max_layers')
        
        encoder_layer_count = random.randint(min_layers, max_layers)
        # Generate architecture code
        architecture_code = ''.join(
            f"{layerCoder.generate_layer_code()}{ENDER}{layerCoder.generate_pooling_layer_code()}{ENDER}"
            for _ in range(encoder_layer_count)
        )
        
        # **** Add head code and enders ****
        architecture_code += f"{layerCoder.generate_head_code(task, encoder_layer_count)}{ENDER}"
        architecture_code += ENDER
        
        # Check if the architecture code is valid
        if self.sanity_checking(architecture_code):
            return architecture_code
        else:
            # Use recursion to generate a new architecture code
            return self.generate_random_architecture_code()

    def architecture2chromosome(self, input_architecture: str) -> List[str]:
        """
        Converts an architecture code into a chromosome (a list of layers) by splitting the architecture code 
        using ENDER as a separator. Additioanlly, handles cases where the architecture ends with one or more enders.

        Parameters:
        input_architecture (str): The architecture code as a string.
        
        Returns:
        List[str]: The chromosome representation of the architecture code as a list of strings.
        """
        # Split the architecture code, remove trailing empty elements
        return [gene for gene in input_architecture.split(ENDER) if gene]

    def chromosome2architecture(self, input_chromosome: List[str]) -> str:
        """
        Converts the chromosome list back into an architecture code, ensuring that it ends with double ender.

        Parameters:
        input_chromosome (List[str]): The chromosome representation as a list of strings.
        
        Returns:
        str: The architecture code as a string.
        """
        return ENDER.join(input_chromosome) + f'{ENDER}{ENDER}'

    def copy(self) -> 'Individual':
        """
        Creates a deep copy of the current individual, preserving its architecture, chromosome, 
        and fitness values.

        Returns:
        Individual: A deep copy of the current individual.
        """
        return copy.deepcopy(self)

    def sanity_checking(self, code: str) -> bool:
        Net = GenericNetwork(
                code, 
                input_channels=3,
                input_height=128,
                input_width=128,
                num_classes=2)
        try:
            Net.build() # Build the network test
        except:
            return False
        
        self.model_size = Net.get_param_size()
        # TODO: config 500 in the config.ini
        if self.model_size < 150:
            print('model size:',self.model_size)
            self.Network = Net
            torch.cuda.empty_cache()
            return True
            # try:
            #     x = torch.randn(1, 3, 128, 128)
            #     self.Network(x)  # Forward pass test
            #     return True
            # except:
            #     return False
            # finally:
            #     # Reset memory
            #     torch.cuda.empty_cache()
            #     return False
        else:
            return False


    # Private methods:
    @property
    def config(self) -> ConfigParser:
        config = ConfigParser()
        config.read(self.config_path)
        return config

    @property
    def fitness(self) -> float:
        return self._fitness
    
    @fitness.setter
    def fitness(self, value: float):
        if value < 0:
            raise ValueError("Fitness value must be non-negative.")
        else:
            self._fitness = value



if __name__ == "__main__":
    pass