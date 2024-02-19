import random


class Wolf:
    def __init__(self, search_space, seed):
        self.rnd = random.Random(seed)
        self.ss = search_space
        self.position = [self.rnd.uniform(min_val, max_val) for min_val, max_val in self.ss]
        self.fitness = 0.0
