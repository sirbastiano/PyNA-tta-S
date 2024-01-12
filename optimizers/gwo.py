# GWO.py
import random
import os
import matplotlib.pyplot as plt
import numpy as np
from classes.wolf import Wolf
from functions.utils import show_wolf_population, calculate_statistics
from functions.fitness import compute_fitness_value_nas as compute_fitness_value


def gwo_optimizer(architecture, search_space, max_iter, n_wolves, logs_directory):
    search_space_keys = list(search_space.keys())
    search_space_values = list(search_space.values())
    dim = len(search_space_values)

    # Logging initialization
    mean_fitness_vector = np.zeros(shape=(max_iter + 1))
    median_fitness_vector = np.zeros_like(mean_fitness_vector)
    best_fitness_vector = np.zeros_like(mean_fitness_vector)

    # Initialize the wolf population
    population = []
    print("START:")
    for i in range(n_wolves):
        temp_wolf = Wolf(search_space_values, seed=7 * i)
        temp_full_position = temp_wolf.position
        temp_wolf.fitness = compute_fitness_value(
            architecture=architecture,
            position=temp_wolf.position,
            keys=search_space_keys,
        )
        population.append(temp_wolf)
        print("position:", temp_wolf.position)
        print("fitness:", temp_wolf.fitness, "\n")

    # Sort the population so that the alpha is the first, then the beta, then the delta, then all the omegas
    population = sorted(population, key=lambda temp: temp.fitness, reverse=True)
    alpha_wolf, beta_wolf, delta_wolf = population[0], population[1], population[2]

    # Logging
    show_wolf_population(population, search_space, iteration=0, logs_dir=logs_directory)
    mean_fitness_vector[0], median_fitness_vector[0] = calculate_statistics(population, attribute='fitness')
    best_fitness_vector[0] = alpha_wolf.fitness

    # Iterations
    rnd = random.Random(0)
    t = 1
    while t <= max_iter:
        # after every iteration, print iteration number and best fitness value so far
        print("Iter = " + str(t) + " best fitness = %.3f" % alpha_wolf.fitness)

        # linearly decreased from 2 to 0
        a = 2 * (1 - t / max_iter)

        for w in population:
            # Update the position of each wolf, exploiting the positions of the three best wolves
            # A = 2 * a * r1 - a
            # C = 2 * r2
            A1 = 2 * a * rnd.random() - a
            A2 = 2 * a * rnd.random() - a
            A3 = 2 * a * rnd.random() - a

            C1 = 2 * rnd.random()
            C2 = 2 * rnd.random()
            C3 = 2 * rnd.random()

            X1 = [0.0 for i in range(dim)]
            X2 = [0.0 for i in range(dim)]
            X3 = [0.0 for i in range(dim)]
            Xnew = [0.0 for i in range(dim)]

            for j in range(dim):
                # Encircling the prey
                # D = abs(C * Xp(t) - X(t))
                # X(t+1) = Xp(t) - A*D
                X1[j] = alpha_wolf.position[j] - A1 * abs(C1 * alpha_wolf.position[j] - w.position[j])
                X2[j] = beta_wolf.position[j] - A2 * abs(C2 * beta_wolf.position[j] - w.position[j])
                X3[j] = delta_wolf.position[j] - A3 * abs(C3 * delta_wolf.position[j] - w.position[j])
                Xnew[j] = (X1[j] + X2[j] + X3[j]) / 3

            # Sanity check is Xnew goes out of search space and clamping
            for i, (min_val, max_val) in enumerate(search_space_values):
                original_value = Xnew[i]
                Xnew[i] = max(min(Xnew[i], max_val), min_val)

                # Check if clamping occurred
                if original_value != Xnew[i]:
                    print("This wolf tried to escape and got CLAMPED!")

            # fitness calculation of new solution
            fnew = compute_fitness_value(
                architecture=architecture,
                position=Xnew,
                keys=search_space_keys,
            )

            # greedy selection
            if fnew >= w.fitness:
                w.position = Xnew
                w.fitness = fnew
                print("This wolf found the scent!")
            else:
                print("This wolf still searches...")

        # Update Xalpha, Xbeta, Xdelta
        population = sorted(population, key=lambda temp: temp.fitness, reverse=True)
        alpha_wolf, beta_wolf, delta_wolf = population[0], population[1], population[2]

        # Logging
        show_wolf_population(population, search_space, iteration=t, logs_dir=logs_directory)
        mean_fitness_vector[t], median_fitness_vector[t] = calculate_statistics(population, attribute='fitness')
        best_fitness_vector[t] = alpha_wolf.fitness

        # Update iteration counter
        t += 1

    # Plot statistics
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
        "position": alpha_wolf.position,
        "fitness": alpha_wolf.fitness
    }

    return best_fit
