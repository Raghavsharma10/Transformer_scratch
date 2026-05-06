def random_offspring(self):
        "Returns an offspring with the associated weight(s)"
        function_set = self.function_set
        function_selection = self._function_selection_ins
        function_selection.density = self.population.density
        function_selection.unfeasible_functions.clear()
        for i in range(self._number_tries_feasible_ind):
            if self._function_selection:
                func_index = function_selection.tournament()
            else:
                func_index = function_selection.random_function()
            func = function_set[func_index]
            args = self.get_args(func)
            if args is None:
                continue
            args = [self.population.population[x].position for x in args]
            f = self._random_offspring(func, args)
            if f is None:
                function_selection.unfeasible_functions.add(func_index)
                continue
            function_selection[func_index] = f.fitness
            return f
        raise RuntimeError("Could not find a suitable random offpsring")