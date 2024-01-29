import random


class Particle:
    def __init__(self, search_space, seed):
        self.rnd = random.Random(seed)
        self.ss = search_space

        self.current_position = [self.rnd.uniform(min_val, max_val) for min_val, max_val in self.ss]
        self.best_position = self.current_position
        self.current_velocity = [self.rnd.gauss(0, 0.1) for _ in range(len(self.current_position))]
        self.current_fitness = 0.0
        self.best_fitness = self.current_fitness