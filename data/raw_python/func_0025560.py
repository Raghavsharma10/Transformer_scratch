def fit(self, X, y, test_set=None):
        """Evolutive process"""
        self._init_time = time.time()
        self.X = X
        if self._popsize == "nvar":
            self._popsize = self.nvar + len(self._input_functions)
        if isinstance(test_set, str) and test_set == 'shuffle':
            test_set = self.shuffle_tr2ts()
        nclasses = self.nclasses(y)
        if self.classifier and self._multiple_outputs:
            pass
        elif nclasses > 2:
            assert False
            self._multiclass = True
            return self.multiclass(X, y, test_set=test_set)
        self.y = y
        if test_set is not None:
            self.Xtest = test_set
        for _ in range(self._number_tries_feasible_ind):
            self._logger.info("Starting evolution")
            try:
                self.create_population()
                if self.stopping_criteria_tl():
                    break
            except RuntimeError as err:
                self._logger.info("Done evolution (RuntimeError (%s), hist: %s)" % (err, len(self.population.hist)))
                return self
            self._logger.info("Population created (hist: %s)" % len(self.population.hist))
            if len(self.population.hist) >= self._tournament_size:
                break
        if len(self.population.hist) == 0:
            raise RuntimeError("Could not find a suitable individual")
        if len(self.population.hist) < self._tournament_size:
            self._logger.info("Done evolution (hist: %s)" % len(self.population.hist))
            return self
        if self._remove_raw_inputs:
            for x in range(self.nvar):
                self._X[x] = None
        while not self.stopping_criteria():
            try:
                a = self.random_offspring()
            except RuntimeError as err:
                self._logger.info("Done evolution (RuntimeError (%s), hist: %s)" % (err, len(self.population.hist)))
                return self
            self.replace(a)
        self._logger.info("Done evolution (hist: %s)" % len(self.population.hist))
        return self