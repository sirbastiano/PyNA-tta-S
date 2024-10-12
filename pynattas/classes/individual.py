from ..functions import architecture_builder as builder
from typing import List


class Individual:
    def __init__(self, max_layers: int):
        """
        Initializes an individual with a random architecture code, converts it to a chromosome, 
        and sets the fitness to zero.
        
        Parameters:
        max_layers (int): Maximum number of layers for the architecture.
        """
        self.architecture: str = builder.generate_random_architecture_code(max_layers=max_layers)
        self.chromosome: List[str] = self.architecture2chromosome(input_architecture=self.architecture)
        self.fitness: float = 0.0

    def architecture2chromosome(self, input_architecture: str) -> List[str]:
        """
        Converts an architecture code into a chromosome list by splitting the architecture code 
        using 'E' as a separator. Handles cases where the architecture ends with one or more 'E's.

        Parameters:
        input_architecture (str): The architecture code as a string.
        
        Returns:
        List[str]: The chromosome representation as a list of strings.
        """
        # Split the architecture code on 'E', remove trailing empty elements
        return [gene for gene in input_architecture.split('E') if gene]

    def chromosome2architecture(self, input_chromosome: List[str]) -> str:
        """
        Converts the chromosome list back into an architecture code, ensuring that it ends with 'EE'.

        Parameters:
        input_chromosome (List[str]): The chromosome representation as a list of strings.
        
        Returns:
        str: The architecture code as a string.
        """
        return 'E'.join(input_chromosome) + 'EE'

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