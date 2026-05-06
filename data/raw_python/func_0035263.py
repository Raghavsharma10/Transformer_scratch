def _reserved_symbols(self):
        """
        Helper property for the build_remap_symbols method.  This
        property first resolves _all_ local references from parents,
        skipping all locally declared symbols as the goal is to generate
        a local mapping for them, but in a way not to shadow over any
        already declared symbols from parents, and also the implicit
        globals in all children.

        This is marked "private" as there are a number of computations
        involved, and is really meant for the build_remap_symbols to use
        for its standard flow.
        """

        # In practice, and as a possible optimisation, the parent's
        # remapped symbols table can be merged into this instance, but
        # this bloats memory use and cause unspecified reservations that
        # may not be applicable this or any child scope.  So for clarity
        # and purity of references made, this somewhat more involved way
        # is done instead.
        remapped_parents_symbols = {
            self.resolve(v) for v in self.non_local_symbols}

        return (
            # block implicit children globals.
            self.global_symbols_in_children |
            # also not any global symbols
            self.global_symbols |
            # also all remapped parent symbols referenced here
            remapped_parents_symbols
        )