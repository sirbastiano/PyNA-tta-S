from pynattas.builder.individual import Individual

class Population:
    """
    A class to represent a population of individuals in a neural architecture search.
    Attributes:
        iteration (int): Current iteration number.
        population (list): List of individuals in the population.
    Methods:
        __init__(max_layers: int, n_individuals: int):
            Initializes the population with a specified number of individuals.
        remove_duplicates(population: list, max_layers: int) -> list:
        store_code(generation: int, logs_directory: str):
        __repr__() -> str:
            Returns a string representation of the Population object.
        __str__() -> str:
            Returns a human-readable string representation of the Population object.
        __len__() -> int:
            Returns the number of individuals in the population.
        __getitem__(key: int):
            Returns the individual at the specified index.
        __setitem__(key: int, value):
            Sets the individual at the specified index.
        __delitem__(key: int):
            Deletes the individual at the specified index.
        __iter__():
            Returns an iterator for the population.
    """
    
    def __init__(self, n_individuals: int, config_path: str = None):
        """
        Ensures the population size is at least n_individuals by adding new individuals if necessary.
        
        Parameters:
        max_layers (int): Maximum number of layers for the architecture.
        n_individuals (int): Number of individuals in the population.
        """
        self.iteration = 0 # Current iteration number
        self.population = []
        
         # Ensure population size is at least n_individuals
        while len(self.population) < n_individuals:
            new_individual = Individual()
            if new_individual.architecture not in [i.architecture for i in self.population]:
                self.population.append(new_individual)
        
        
    def remove_duplicates(self) -> None:
        """
        Removes duplicate individuals from the population.
        
        Parameters:
        population (list): The population of individuals.
        max_layers (int): Maximum number of layers for the architecture.
        """
        unique_population = []
        for individual in self.population:
            if individual.architecture not in [i.architecture for i in unique_population]:
                unique_population.append(individual)
        self.population = unique_population    
    
    def store_code(self, generation: int, logs_directory: str):
        """
        Stores the population in a file for the current generation, including fitness values.
        
        Parameters:
        generation (int): The current generation number.
        logs_directory (str): The directory for log and plot files.
        """
        with open(f"{logs_directory}/population_{generation}.txt", 'w') as file:
            for individual in self.population:
                file.write(f"Architecture: {individual.architecture}, Fitness: {individual.fitness}\n")


    def __repr__(self):
        return f"Population(size={len(self.population)}, iteration={self.iteration})"


    def __str__(self):
        return f"Population with {len(self.population)} individuals at iteration {self.iteration}"


    def __len__(self):
        return len(self.population)
    
    
    def __getitem__(self, key):
        return self.population[key]
    
    
    def __setitem__(self, key, value):
        self.population[key] = value
        
    
    def __delitem__(self, key):
        del self.population[key]
        
    
    def __iter__(self):
        return iter(self.population)


if __name__ == '__main__':
    pop = Population(max_layers=5, n_individuals=10)
    print(len(pop.population))
    for idx, guy in enumerate(pop.population):
        print(idx, guy.architecture)
        
    pop.store(generation=0, logs_directory='logs/GA_logs')