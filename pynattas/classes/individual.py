import random
import functions.architecture_builder as builder


class Individual:
    def __init__(self, max_layers):
        self.architecture = builder.generate_random_architecture_code(max_layers=max_layers)
        self.chromosome = self.architecture2chromosome(self.architecture)
        self.fitness = 0.0

    def architecture2chromosome(self, input_architecture):
        """
        Converts an architecture code into a chromosome list.

        Example:
        Input: "c1gc1rm1rMC"
        Output: ["c1g", "c1r", "m1r", "M", "C"]
        """
        # Initialize an empty chromosome list
        chromosome = []

        # Loop through the architecture code to extract genes
        i = 0
        while i < len(input_architecture):
            # Check if at the last two characters (pooling layer and head)
            if i >= len(input_architecture) - 2:
                chromosome.append(input_architecture[i])
                i += 1
            else:
                # Extract the triplet and add it as a gene
                gene = input_architecture[i:i+3]
                chromosome.append(gene)
                i += 3

        return chromosome

    def chromosome2architecture(self, chromosome):
        """
            Converts chromosome list into an architecture code.
        """
        architecture = ""
        for gene in chromosome:
            architecture += gene

        return architecture

    def copy(self):
        """
        Creates a deep copy of the current individual.

        Returns:
        Individual: A new Individual object with the same architecture, chromosome, and fitness.
        """
        new_individual = Individual(max_layers=len(self.chromosome))
        new_individual.architecture= self.architecture
        new_individual.chromosome = self.chromosome.copy()  # Ensure a deep copy of the chromosome list
        new_individual.fitness = self.fitness

        return new_individual