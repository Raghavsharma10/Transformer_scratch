def global_symbols(self):
        """
        These are symbols that have been referenced, but not declared
        within this scope or any parent scopes.
        """

        declared_symbols = self.declared_symbols
        return set(
            s for s in self.referenced_symbols if s not in declared_symbols)