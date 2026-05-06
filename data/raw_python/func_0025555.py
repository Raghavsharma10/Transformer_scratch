def population(self):
        "Class containing the population and all the individuals generated"
        try:
            return self._p
        except AttributeError:
            self._p = self._population_class(base=self,
                                             tournament_size=self._tournament_size,
                                             classifier=self.classifier,
                                             labels=self._labels,
                                             es_extra_test=self.es_extra_test,
                                             popsize=self._popsize,
                                             random_generations=self._random_generations,
                                             negative_selection=self._negative_selection)
            return self._p