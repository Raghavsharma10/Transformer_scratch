def add_new_reset_method(obj):
    """
    Attach a new `reset()` method to `obj` which resets the internal
    seed generator of `obj` and then resets each of its constituent
    field generators found in `obj.field_gens`.
    """

    #
    # Create and assign automatically generated reset() method
    #


    def new_reset(self, seed=None):
        logger.debug(f'[EEE] Inside automatically generated reset() method for {self} (seed={seed})')

        if seed is not None:
            self.seed_generator.reset(seed)
            for name, gen in self.field_gens.items():
                next_seed = next(self.seed_generator)
                gen.reset(next_seed)

            # TODO: the following should be covered by the newly added
            # reset() method in IndependentGeneratorMeta. However, for
            # some reason we can't call this via the usual `orig_reset()`
            # pattern, so we have to duplicate this here. Not ideal...
            for c in self._dependent_generators:
                c.reset_dependent_generator(seed)

        return self

    obj.reset = new_reset