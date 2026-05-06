def close(self):
        """
        Mark the scope as closed, i.e. all symbols have been declared,
        and no further declarations should be done.
        """

        if self._closed:
            raise ValueError('scope is already marked as closed')

        # By letting parent know which symbols this scope has leaked, it
        # will let them reserve all lowest identifiers first.
        if self.parent:
            for symbol, c in self.leaked_referenced_symbols.items():
                self.parent.reference(symbol, c)

        self._closed = True