def lookup(cls, basenote):
        """Look up note in registered annotations, walking class tree."""
        # Walk method resolution order, which includes current class.
        for c in cls.mro():
            if 'provider_registry' not in vars(c):
                # class is a mixin, super to base class, or never registered.
                continue
            if basenote in c.provider_registry:
                # note is in the registry.
                return c.provider_registry[basenote]
        raise LookupError(repr(basenote))