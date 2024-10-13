import random
from pynattas.utils import layerCoder
from typing import List, Optional

ENDER = layerCoder.ARCHITECTURE_ENDER

class Individual:
    def __init__(self, max_layers: int):
        """
        Initializes an individual with a random architecture code, converts it to a chromosome, 
        and sets the fitness to zero.
        
        Parameters:
        max_layers (int): Maximum number of layers for the architecture.
        """
        self.architecture: str = self.generate_random_architecture_code(max_layers=max_layers)
        self.chromosome: List[str] = self.architecture2chromosome(input_architecture=self.architecture)
        self.fitness: float = 0.0

    def generate_random_architecture_code(self, max_layers: int, config_path: Optional[str] = 'config.ini') -> str:
        """
        Generates a random architecture code with a variable number of layers, based on a specified task.

        Parameters:
        max_layers (int): Maximum number of layers for the architecture.
        config_path (Optional[str]): Path to the configuration file (defaults to 'config.ini').

        Returns:
        str: Generated architecture code.
        """
        task = layerCoder.get_task_from_config(config_path)

        min_layers = 3 if task == 'D' else 1
        encoder_layer_count = random.randint(min_layers, max_layers)
        
        # Generate architecture code
        architecture_code = ''.join(
            f"{generate_layer_code()}{ENDER}{generate_pooling_layer_code()}{ENDER}"
            for _ in range(encoder_layer_count)
        )
        
        logger.info(f"This architecture has {encoder_layer_count} encoder layers.")
        
        # **** Add head code and enders ****
        architecture_code += f"{generate_head_code(task, encoder_layer_count)}{ENDER}"
        architecture_code += ENDER

        return architecture_code



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
        new_individual = Individual(max_layers=len(self.chromosome))
        new_individual.architecture = self.architecture
        new_individual.chromosome = self.chromosome.copy()  # Deep copy the list
        new_individual.fitness = self.fitness
        return new_individual



if __name__ == "__main__":
    pass