from pynattas.builder.individual import Individual
from pynattas.builder.netBuilder import GenericNetwork


class Population:
    def __init__(self, max_layers: int, n_individuals: int):
        """
        self.population = self.remove_duplicates(population=self.population, max_layers=max_layers)
        self.ensure_population_size(n_individuals=n_individuals, max_layers=max_layers)
        and sets the fitness to zero.
        
        Parameters:
        max_layers (int): Maximum number of layers for the architecture.
        n_individuals (int): Number of individuals in the population.
        """
        self.iteration = 0 # Current iteration number
        self.population = []
        for i in range(n_individuals):
            temp_individual = Individual(max_layers=max_layers)
            self.population.append(temp_individual)
        
        while len(self.population) < n_individuals: # Ensure population size is at least n_individuals
            self.population = self.remove_duplicates(population=self.population, max_layers=max_layers)
            self.ensure_population_size(n_individuals=n_individuals, max_layers=max_layers)
        
    def remove_duplicates(self, population: list, max_layers: int) -> list:
        """
        Removes duplicate individuals from the population.
        
        Parameters:
        population (list): The population of individuals.
        max_layers (int): Maximum number of layers for the architecture.
        
        Returns:
        list: The population with duplicates removed.
        """
        unique_population = []
        for individual in population:
            if individual.architecture not in [i.architecture for i in unique_population]:
                unique_population.append(individual)
        return unique_population
    
    
    def ensure_population_size(self, n_individuals: int, max_layers: int):
        """
        Ensures the population size is at least n_individuals by adding new individuals if necessary.
        
        Parameters:
        n_individuals (int): The desired number of individuals in the population.
        max_layers (int): Maximum number of layers for the architecture.
        """
        while len(self.population) < n_individuals:
            temp_individual = Individual(max_layers=max_layers)
            if temp_individual.architecture not in [i.architecture for i in self.population]:
                self.population.append(temp_individual)
    
    
    def store(self, generation: int, logs_directory: str):
        """
        Stores the population in a file for the current generation, including fitness values.
        
        Parameters:
        generation (int): The current generation number.
        logs_directory (str): The directory for log and plot files.
        """
        with open(f"{logs_directory}/population_{generation}.txt", 'w') as file:
            for individual in self.population:
                file.write(f"Architecture: {individual.architecture}, Fitness: {individual.fitness}\n")

if __name__ == '__main__':
    pop = Population(max_layers=5, n_individuals=10)
    print(len(pop.population))
    for idx, guy in enumerate(pop.population):
        print(idx, guy.architecture)
        
    pop.store(generation=0, logs_directory='logs/GA_logs')