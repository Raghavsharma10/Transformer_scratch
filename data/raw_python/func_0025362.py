def create_population(self, popsize=1000, min_depth=2,
                          max_depth=4,
                          X=None):
        "Creates random population using ramped half-and-half method"
        import itertools
        args = [x for x in itertools.product(range(min_depth,
                                                   max_depth+1),
                                             [True, False])]
        index = 0
        output = []
        while len(output) < popsize:
            depth, full = args[index]
            index += 1
            if index >= len(args):
                index = 0
            if full:
                ind = self.create_random_ind_full(depth=depth)
            else:
                ind = self.create_random_ind_grow(depth=depth)
            flag = True
            if X is not None:
                x = Individual(ind)
                x.decision_function(X)
                flag = x.individual[0].isfinite()
            l_vars = (flag, len(output), full, depth, len(ind))
            l_str = " flag: %s len(output): %s full: %s depth: %s len(ind): %s"
            self._logger.debug(l_str % l_vars)
            if flag:
                output.append(ind)
        return output