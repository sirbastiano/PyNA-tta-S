# GA.py
import random
import matplotlib.pyplot as plt
import os
import numpy as np
from classes.individual import Individual
from functions.utils import show_population, calculate_statistics
from functions.fitness import compute_fitness_value_nas as compute_fitness_value


def single_point_crossover(parents, crossover_cutoff):
    k = crossover_cutoff
    children = parents.copy()

    temp_00 = parents[0].chromosome[:k].copy()
    temp_01 = parents[1].chromosome[k:].copy()
    temp_10 = parents[1].chromosome[:k].copy()
    temp_11 = parents[0].chromosome[k:].copy()

    children[0].chromosome = temp_00 + temp_01
    children[1].chromosome = temp_10 + temp_11

    return children


def mutation(children, search_space, n_genes, mutation_probability):
    for child in children:
        for gene_i in range(n_genes):
            rnd = random.random()
            if rnd <= mutation_probability:
                child.chromosome[gene_i] = random.uniform(search_space[gene_i][0], search_space[gene_i][1])
    return children


def ga_optimizer(architecture, search_space, max_iter, n_individuals, mating_pool_cutoff, crossover_cutoff, mutation_probability, logs_directory):
    search_space_keys = search_space.keys()
    search_space_values = list(search_space.values())
    dim = len(search_space_values)

    # Sanity checks
    if n_individuals % 2 == 1:
        print("ERROR: population_size should be an even number.")
        exit()
    elif mating_pool_cutoff > 1.0:
        print("ERROR: mating_pool_cutoff should be less than 1.")
        exit()
    elif crossover_cutoff >= dim:
        print("ERROR: crossover_cutoff should be less than the length of a chromosome.")
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
        temp_individual = Individual(search_space_values, seed=2 * i)
        population.append(temp_individual)

    print("Starting chromosome pool:")
    for i in population:
        i.fitness = compute_fitness_value(position=i.chromosome, keys=search_space_keys, architecture=architecture)
        print(f"Individual {i}")
        print(f"chromosome: {i.chromosome}")
        print(f"fitness: {i.fitness}\n")

    # Starting population update
    population = sorted(population, key=lambda temp: temp.fitness, reverse=True)
    historical_best_fitness = population[0].fitness
    fittest_individual = population[0].chromosome
    mean_fitness_vector[0], median_fitness_vector[0] = calculate_statistics(population, attribute='fitness')
    best_fitness_vector[0] = historical_best_fitness

    show_population(
        population=population,
        search_space=search_space_values,
        generation=0,
        logs_dir=logs_directory,
        historical_best_fitness=historical_best_fitness,
        fittest_individual=fittest_individual,
    )

    # Iterations
    t = 1
    while t <= max_iter:
        print(f"*** GENERATION {t} ***")
        new_population = []

        # Create a mating pool
        mating_pool = population[:int(np.floor(mating_pool_cutoff * len(population)))].copy()
        for i in range(int(np.ceil((1 - mating_pool_cutoff) * len(population)))):
            temp_individual = Individual(search_space_values, seed=12121 + i)
            mating_pool.append(temp_individual)

        # Coupling and mating
        couple_i = 0
        while couple_i < len(mating_pool):
            parents = [mating_pool[couple_i], mating_pool[couple_i + 1]]
            children = single_point_crossover(parents=parents, crossover_cutoff=crossover_cutoff)
            children = mutation(children=children, search_space=search_space_values, n_genes=dim, mutation_probability=mutation_probability)
            new_population = new_population + children
            couple_i += 2

        # Update the population
        population = new_population.copy()
        for i in population:
            i.fitness = compute_fitness_value(position=i.chromosome, keys=search_space_keys, architecture=architecture)
            print("position", i.chromosome)
            print("fitness:", i.fitness, "\n")

        # Update historical best
        population = sorted(population, key=lambda temp: temp.fitness, reverse=True)
        if historical_best_fitness <= population[0].fitness:
            historical_best_fitness = population[0].fitness
            fittest_individual = population[0].chromosome

        if t == max_iter:
            print(f"THE LAST GENERATION ({t}):")
        print(f"For generation {t}, the best fitness of the population is {population[0].fitness}.")
        print(f"The best historical fitness is {historical_best_fitness},"
              f"with the most fit individual having the following genes: {fittest_individual}.")

        show_population(
            population=population,
            search_space=search_space_values,
            generation=t,
            logs_dir=logs_directory,
            historical_best_fitness=historical_best_fitness,
            fittest_individual=fittest_individual,
        )

        # Update analytics
        mean_fitness_vector[t], median_fitness_vector[t] = calculate_statistics(population, attribute='fitness')
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
        "fitness": historical_best_fitness
    }

    return best_fit
