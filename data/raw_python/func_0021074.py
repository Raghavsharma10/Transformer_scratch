def append(self, value):
        """Add *value* to the right side of the collection."""
        def append_trans(pipe):
            self._append_helper(value, pipe)

        self._transaction(append_trans)