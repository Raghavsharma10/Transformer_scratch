def reset_input_generators(self, seed):
        """
        Helper method which explicitly resets all input generators
        to the derived generator. This should only ever be called
        for testing or debugging.
        """
        seed_generator = SeedGenerator().reset(seed=seed)

        for gen in self.input_generators:
            gen.reset(next(seed_generator))
            try:
                # In case `gen` is itself a derived generator,
                # recursively reset its own input generators.
                gen.reset_input_generators(next(seed_generator))
            except AttributeError:
                pass