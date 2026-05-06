def global_symbols_in_children(self):
        """
        This is based on all children referenced symbols that have not
        been declared.

        The intended use case is to ban the symbols from being used as
        remapped symbol values.
        """

        result = set()
        for child in self.children:
            result |= (
                child.global_symbols |
                child.global_symbols_in_children)
        return result