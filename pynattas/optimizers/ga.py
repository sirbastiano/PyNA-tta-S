# GA.py
import random
import matplotlib.pyplot as plt
import os
import numpy as np
from pynattas import builder
from pynattas.utils import layerCoder


def single_point_crossover(parents):
    """
    Performs a single-point crossover between two parent chromosomes.

    Parameters:
    parents (list): The parent chromosomes as lists of genes.

    Returns:
    list: A list containing two new child chromosomes resulting from the crossover.
    The crossover point is randomly selected within the range of the shorter chromosome length.
    """

    # Determine the length of the shorter parent chromosome
    min_length = min(len(parents[0].chromosome), len(parents[1].chromosome))

    # Randomly select a crossover point, ensuring it is within the range of both chromosomes
    crossover_cutoff = random.randint(1, min_length - 2)
    k = crossover_cutoff

    temp_00 = parents[0].chromosome[:k].copy()
    temp_01 = parents[1].chromosome[k:].copy()
    temp_10 = parents[1].chromosome[:k].copy()
    temp_11 = parents[0].chromosome[k:].copy()

    # Perform crossover
    children = parents.copy()
    children[0].chromosome = temp_00 + temp_01
    children[1].chromosome = temp_10 + temp_11

    return children


def mutation(children, mutation_probability):
    """
    Apply mutation to a list of child chromosomes.

    Parameters:
    children (list): A list of child chromosomes.
    mutation_probability (float): The probability of a gene mutation.

    Returns:
    list: The mutated child chromosomes.
    """
    for child in children:
        for gene_index in range(len(child.chromosome)):
            rnd = random.random()
            if rnd <= mutation_probability:
                gene = child.chromosome[gene_index]
                # Mutate based on the type of gene
                if gene[0]=='L':  # Backbone layers
                    child.chromosome[gene_index] = layerCoder.generate_layer_code()
                elif gene[0]=='P': # Pooling layer gene
                    child.chromosome[gene_index] = layerCoder.generate_pooling_layer_code()
                elif gene[0]=='H': # Head gene
                    break
                else:
                    print("Something went wrong with mutation.")
                    exit()
    return children


def remove_duplicates(population, max_layers):
    """
    Remove duplicates from the population by replacing them with unique individuals.

    Parameters:
    population (list): A list of individuals in the population.
    max_layers (int): The maximum number of layers for an individual.

    Returns:
    list: The updated population with duplicates removed.
    """
    unique_architectures = set()
    updated_population = []

    for individual in population:
        if individual.architecture not in unique_architectures:
            unique_architectures.add(individual.architecture)
            updated_population.append(individual)
        else:
            # Attempt to generate a unique individual up to 50 times
            for _ in range(50):
                new_individual = classes.Individual(max_layers=max_layers)
                if new_individual.architecture not in unique_architectures:
                    unique_architectures.add(new_individual.architecture)
                    updated_population.append(new_individual)
                    break

    return updated_population


def ga_optimizer(max_layers, max_iter, n_individuals, mating_pool_cutoff, mutation_probability, logs_directory, task):
    """
    Genetic Algorithm optimizer for architecture search.

    Parameters:
    max_layers (int): The maximum number of convolutional modules for a generated individual.
    max_iter (int): The maximum number of iterations/generations.
    n_individuals (int): The number of individuals in each generation.
    mating_pool_cutoff (float): The percentage of the population used for mating.
    mutation_probability (float): The probability of gene mutation.
    logs_directory (str): The directory for log and plot files.

    Returns:
    dict: A dictionary containing the best architecture and its fitness.
    """

    # Sanity checks
    if max_layers < 1:
        print("Error: Max layers should be bigger than 0.")
        exit()
    elif n_individuals % 2 == 1:
        print("ERROR: population_size should be an even number.")
        exit()
    elif mating_pool_cutoff > 1.0:
        print("ERROR: mating_pool_cutoff should be less than 1.")
        exit()
    elif mutation_probability > 1.0:
        print("ERROR: mutation_probability should be less than 1.")
        exit()

    # Logging initialization
    mean_fitness_vector = np.zeros(shape=(max_iter + 1))
    median_fitness_vector = np.zeros_like(mean_fitness_vector)
    best_fitness_vector = np.zeros_like(mean_fitness_vector)

    # Population Initialization
    population = []
    for i in range(n_individuals):
        temp_individual = classes.Individual(max_layers=max_layers)
        population.append(temp_individual)

    population = remove_duplicates(population=population, max_layers=max_layers)

    print("Starting chromosome pool:")
    for i in population:
        parsed_layers = layerCoder.parse_architecture_code(i.architecture)
        print(f"Architecture: {i.architecture}")
        print(f"Chromosome: {i.chromosome}")   

        if task == 'D':
            i.fitness = fitness.compute_fitness_value_OD(parsed_layers=parsed_layers)
        else:
            i.fitness = fitness.compute_fitness_value(parsed_layers=parsed_layers)
        
        print(f"chromosome: {i.chromosome}, fitness: {i.fitness}\n")

    # Starting population update
    population = sorted(population, key=lambda temp: temp.fitness, reverse=True)
    historical_best_fitness = population[0].fitness
    fittest_individual = population[0].architecture
    fittest_genes = population[0].chromosome
    mean_fitness_vector[0], median_fitness_vector[0] = utils.calculate_statistics(population, attribute='fitness')
    best_fitness_vector[0] = historical_best_fitness

    utils.show_population(
        population=population,
        generation=0,
        logs_dir=logs_directory,
        historical_best_fitness=historical_best_fitness,
        fittest_individual=fittest_individual
    )

    # Iterations
    t = 1
    while t <= max_iter:
        print(f"*** GENERATION {t} ***")
        new_population = []

        # Create a mating pool
        mating_pool = population[:int(np.floor(mating_pool_cutoff * len(population)))].copy()
        for i in range(int(np.ceil((1 - mating_pool_cutoff) * len(population)))):
            temp_individual = classes.Individual(max_layers=max_layers)
            mating_pool.append(temp_individual)

        # Coupling and mating
        couple_i = 0
        while couple_i < len(mating_pool):
            parents = [mating_pool[couple_i], mating_pool[couple_i + 1]]
            children = single_point_crossover(parents=parents)
            children = mutation(
                children=children,
                mutation_probability=mutation_probability,
            )
            new_population = new_population + children
            couple_i += 2

        # Update the population
        population = new_population.copy()
        for i in population:
            i.architecture = i.chromosome2architecture(i.chromosome)
        population = remove_duplicates(population=population, max_layers=max_layers)

        for i in population:
            parsed_layers = layerCoder.parse_architecture_code(i.architecture)
            print(f"Architecture: {i.architecture}")
            print(f"Chromosome: {i.chromosome}")
            if task == 'D':
                i.fitness = fitness.compute_fitness_value_OD(parsed_layers=parsed_layers)
            else:
                i.fitness = fitness.compute_fitness_value(parsed_layers=parsed_layers)
            print(f"chromosome: {i.chromosome}, fitness: {i.fitness}\n")

        # Update historical best
        population = sorted(population, key=lambda temp: temp.fitness, reverse=True)
        if historical_best_fitness <= population[0].fitness:
            historical_best_fitness = population[0].fitness
            fittest_individual = population[0].architecture
            fittest_genes = population[0].chromosome

        if t == max_iter:
            print(f"THE LAST GENERATION ({t}):")
        print(f"For generation {t}, the best fitness of the population is {population[0].fitness}.")
        print(f"The best historical fitness is {historical_best_fitness},"
              f"with the most fit individual having the following genes: {fittest_genes}.")

        utils.show_population(
            population=population,
            generation=t,
            logs_dir=logs_directory,
            historical_best_fitness=historical_best_fitness,
            fittest_individual=fittest_individual,
        )

        # Update analytics
        mean_fitness_vector[t], median_fitness_vector[t] = utils.calculate_statistics(population, attribute='fitness')
        best_fitness_vector[t] = historical_best_fitness

        t += 1

    plt.figure(figsize=(12, 8))

    iterations = np.arange(0, max_iter + 1, 1)
    plt.plot(iterations, mean_fitness_vector, color='red', label='mean')
    plt.plot(iterations, median_fitness_vector, color='blue', label='median')
    plt.plot(iterations, best_fitness_vector, color='green', label='best')

    plt.xlabel("Iteration")
    plt.ylabel("")
    plt.grid()
    plt.title("Mean, Median, and Historical Best fitness over the iterations")
    plt.legend()

    plt.savefig(os.path.join(logs_directory, f'plot_stats_over_iterations.png'), bbox_inches='tight')
    # plt.show()

    # The final result is the position of the alpha
    best_fit = {
        "position": fittest_individual,
        "fitness": historical_best_fitness,
    }

    return best_fit
