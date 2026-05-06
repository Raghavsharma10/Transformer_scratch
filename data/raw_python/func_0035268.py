def build_remap_symbols(self, name_generator, children_only=None):
        """
        The children_only flag is inapplicable, but this is included as
        the Scope class is defined like so.

        Here this simply just place the catch symbol with the next
        replacement available.
        """

        replacement = name_generator(skip=(self._reserved_symbols))
        self.remapped_symbols[self.catch_symbol] = next(replacement)

        # also to continue down the children.
        for child in self.children:
            child.build_remap_symbols(name_generator, False)