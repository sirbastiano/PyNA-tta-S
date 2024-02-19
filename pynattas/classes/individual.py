from ..functions import architecture_builder as builder


class Individual:
    def __init__(self, max_layers):
        self.architecture = builder.generate_random_architecture_code(max_layers=max_layers)
        self.chromosome = self.architecture2chromosome(input_architecture=self.architecture)
        self.fitness = 0.0

    def architecture2chromosome(self, input_architecture):
        """
        Converts an architecture code into a chromosome list by splitting
        the architecture code using 'E'. This method also handles the case where
        the architecture ends with 'EE', avoiding an empty string at the end of the list.
        """
        # Split the architecture code on 'E'
        chromosome = input_architecture.split('E')
        # Remove the last two empty elements if the architecture ends with 'EE'
        if len(chromosome) >= 2 and chromosome[-1] == '' and chromosome[-2] == '':
            chromosome = chromosome[:-2]
        elif len(chromosome) >= 1 and chromosome[-1] == '':
            # If it only ends with a single 'E', just remove the last empty element
            chromosome = chromosome[:-1]
        return chromosome

    def chromosome2architecture(self, input_chromosome):
        """
        Converts the chromosome list back into an architecture code by joining
        the list items with 'E' and ensuring the architecture ends with 'EE'.
        """
        architecture_code = 'E'.join(input_chromosome) + 'EE'
        return architecture_code

    def copy(self):
        """
        Creates a deep copy of the current individual, including architecture,
        chromosome, and fitness.
        """
        new_individual = Individual(max_layers=len(self.chromosome))
        new_individual.architecture = self.architecture
        new_individual.chromosome = self.chromosome.copy()
        new_individual.fitness = self.fitness
        return new_individual
