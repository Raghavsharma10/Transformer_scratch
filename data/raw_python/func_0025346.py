def set_fitness(self, v):
        """Set the fitness to a new node.
        Returns false in case fitness is not finite"""
        base = self._base
        self.fitness(v)
        if not np.isfinite(v.fitness):
            self.del_error(v)
            return False
        if base._tr_fraction < 1:
            self.fitness_vs(v)
            if not np.isfinite(v.fitness_vs):
                self.del_error(v)
                return False
        self.del_error(v)
        return True