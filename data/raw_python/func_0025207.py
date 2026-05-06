def replace(self, v):
        """Replace an individual selected by negative tournament selection with
        individual v"""
        if self.popsize < self._popsize:
            return self.add(v)
        k = self.tournament(negative=True)
        self.clean(self.population[k])
        self.population[k] = v
        v.position = len(self._hist)
        self._hist.append(v)
        self.bsf = v
        self.estopping = v
        self._inds_replace += 1
        self._density += self.get_density(v)
        if self._inds_replace == self._popsize:
            self._inds_replace = 0
            self.generation += 1
            gc.collect()