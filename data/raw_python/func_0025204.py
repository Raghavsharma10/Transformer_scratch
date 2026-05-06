def tournament(self, negative=False):
        """Tournament selection and when negative is True it performs negative
        tournament selection"""
        if self.generation <= self._random_generations and not negative:
            return self.random_selection()
        if not self._negative_selection and negative:
            return self.random_selection(negative=negative)
        vars = self.random()
        fit = [(k, self.population[x].fitness) for k, x in enumerate(vars)]
        if negative:
            fit = min(fit, key=lambda x: x[1])
        else:
            fit = max(fit, key=lambda x: x[1])
        index = fit[0]
        return vars[index]