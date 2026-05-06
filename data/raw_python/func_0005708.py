def add_new_reset_method(cls):
    """
    Replace existing cls.reset() method with a new one which also
    calls reset() on any clones.
    """
    orig_reset = cls.reset

    def new_reset(self, seed=None):
        logger.debug(f"Calling reset() on {self} (seed={seed})")
        orig_reset(self, seed)
        for c in self._dependent_generators:
            c.reset_dependent_generator(seed)
        return self

    cls.reset = new_reset