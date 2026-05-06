def all_generators(self):
        """
        Convenience property to iterate over all generators in arg_gens and kwarg_gens.
        """
        for arg_gen in self.arg_gens:
            yield arg_gen

        for kwarg_gen in self.kwarg_gens.values():
            yield kwarg_gen