def _add_new_reset_method(cls):
    """
    Attach a new `reset()` method to `cls` which resets the internal
    seed generator of `cls` and then resets each of its constituent
    field generators found in `cls.field_gens`.
    """

    #
    # Create and assign automatically generated reset() method
    #


    def new_reset_method(self, seed=None):
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
            for c in self._clones:
                c.reset_clone(seed)

        return self

    cls.reset = new_reset_method