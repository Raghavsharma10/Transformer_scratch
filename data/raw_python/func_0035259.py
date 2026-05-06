def declared_symbols(self):
        """
        Return all local symbols here, and also of the parents
        """

        return self.local_declared_symbols | (
            self.parent.declared_symbols if self.parent else set())