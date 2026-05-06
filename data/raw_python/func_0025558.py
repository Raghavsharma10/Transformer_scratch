def stopping_criteria(self):
        "Test whether the stopping criteria has been achieved."
        if self.stopping_criteria_tl():
            return True
        if self.generations < np.inf:
            inds = self.popsize * self.generations
            flag = inds <= len(self.population.hist)
        else:
            flag = False
        if flag:
            return True
        est = self.population.estopping
        if self._tr_fraction < 1:
            if est is not None and est.fitness_vs == 0:
                return True
        esr = self._early_stopping_rounds
        if self._tr_fraction < 1 and esr is not None and est is not None:
            position = self.population.estopping.position
            if position < self.init_popsize:
                position = self.init_popsize
            return (len(self.population.hist) +
                    self._unfeasible_counter -
                    position) > esr
        return flag