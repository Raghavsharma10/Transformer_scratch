def add(self, v):
        "Add an individual to the population"
        self.population.append(v)
        self._current_popsize += 1
        v.position = len(self._hist)
        self._hist.append(v)
        self.bsf = v
        self.estopping = v
        self._density += self.get_density(v)