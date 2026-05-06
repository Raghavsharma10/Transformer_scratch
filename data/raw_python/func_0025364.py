def fitness_vs(self):
        "Median Fitness in the validation set"
        l = [x.fitness_vs for x in self.models]
        return np.median(l)