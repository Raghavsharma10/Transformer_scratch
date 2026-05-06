def build_remap_symbols(self, name_generator, children_only=True):
        """
        This builds the replacement table for all the defined symbols
        for all the children, and this scope, if the children_only
        argument is False.
        """

        if not children_only:
            replacement = name_generator(skip=(self._reserved_symbols))
            for symbol, c in reversed(sorted(
                    self.referenced_symbols.items(), key=itemgetter(1, 0))):
                if symbol not in self.local_declared_symbols:
                    continue
                self.remapped_symbols[symbol] = next(replacement)

        for child in self.children:
            child.build_remap_symbols(name_generator, False)