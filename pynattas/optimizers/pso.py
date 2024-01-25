# PSO.py
import random
import matplotlib.pyplot as plt
import os
import numpy as np
from classes.particle import Particle
from functions.utils import show_particle_swarm, calculate_statistics
from functions.fitness import compute_fitness_value_nas as compute_fitness_value


def update_global_best(swarm, old_global_best_position, old_global_best_fitness):
    new_global_best_position = old_global_best_position
    new_global_best_fitness = old_global_best_fitness

    for particle in swarm:
        if new_global_best_fitness < particle.current_fitness:
            new_global_best_position = particle.current_position
            new_global_best_fitness = particle.current_fitness

    return new_global_best_position, new_global_best_fitness


def pso_optimizer(architecture, search_space, max_iter, n_particles, c1, c2, w, logs_directory):
    search_space_keys = search_space.keys()
    search_space_values = list(search_space.values())
    dim = len(search_space_values)

    # Logging initialization
    mean_fitness_vector = np.zeros(shape=(max_iter + 1))
    median_fitness_vector = np.zeros_like(mean_fitness_vector)
    best_fitness_vector = np.zeros_like(mean_fitness_vector)

    # Swarm initialization
    swarm = []
    for i in range(n_particles):
        temp_particle = Particle(search_space_values, seed=2 * i)
        temp_particle.current_fitness = compute_fitness_value(
            position=temp_particle.current_position,
            keys=search_space_keys,
            architecture=architecture,
        )
        temp_particle.best_fitness = temp_particle.current_fitness
        swarm.append(temp_particle)

    print("Initial Swarm:")
    for i, particle in enumerate(swarm):
        print(f"Particle {i} - Position: {particle.current_position}")
        print("               Velocity:", particle.current_velocity)
        print("               Fitness:", particle.current_fitness)

    # Global best initialization
    global_best_position = (0.0, 0.0)
    global_best_fitness = -100000.0

    # Global best update
    global_best_position, global_best_fitness = update_global_best(swarm, global_best_position, global_best_fitness)
    best_fitness_vector[0] = global_best_fitness
    mean_fitness_vector[0], median_fitness_vector[0] = calculate_statistics(swarm, attribute='current_fitness')

    show_particle_swarm(
        swarm=swarm,
        search_space=search_space_values,
        global_best_position=global_best_position,
        global_best_fitness=global_best_fitness,
        logs_directory=logs_directory,
        iteration=0,
    )

    # Initialization of terms used during iterations
    inertial_term = np.ones_like(swarm[0].current_position)
    cognitive_term = np.ones_like(swarm[0].current_position)
    social_term = np.ones_like(swarm[0].current_position)

    # Iterations
    t = 1
    while t <= max_iter:
        print(f"\nIteration {t}:")
        r1 = random.random()
        r2 = random.random()

        for particle in swarm:
            for j in range(dim):
                inertial_term[j] = w * particle.current_velocity[j]
                cognitive_term[j] = c1 * r1 * (particle.best_position[j] - particle.current_position[j])
                social_term[j] = c2 * r2 * (global_best_position[j] - particle.current_position[j])
                particle.current_velocity[j] = inertial_term[j] + cognitive_term[j] + social_term[j]

            particle.current_position = [pos + vel for pos, vel in zip(particle.current_position, particle.current_velocity)]

            clamped_position = np.zeros_like(particle.current_position)
            for j in range(dim):
                clamped_position[j] = max(
                    float(search_space_values[j][0]),
                    min(particle.current_position[j], float(search_space_values[j][1])),
                )
                if particle.current_position[j] != clamped_position[j]:
                    particle.current_position[j] = clamped_position[j]
                    print(f"A particle got clamped in dimension {j}.")

            particle.current_fitness = compute_fitness_value(
                position=particle.current_position,
                keys=search_space_keys,
                architecture=architecture,
            )

            if particle.best_fitness < particle.current_fitness:
                particle.best_position = particle.current_position
                particle.best_fitness = particle.current_fitness

        global_best_position, global_best_fitness = update_global_best(swarm, global_best_position, global_best_fitness)
        best_fitness_vector[t] = global_best_fitness
        print(f"Best fitness: {global_best_fitness}, at position: {global_best_position}")

        show_particle_swarm(
            swarm=swarm,
            search_space=search_space_values,
            global_best_position=global_best_position,
            global_best_fitness=global_best_fitness,
            logs_directory=logs_directory,
            iteration=t
        )

        mean_fitness_vector[t], median_fitness_vector[t] = calculate_statistics(swarm, attribute='current_fitness')
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
    #plt.show()

    # The final result is the position of the alpha
    best_fit = {
        "position": global_best_position,
        "fitness": global_best_fitness
    }

    return best_fit
