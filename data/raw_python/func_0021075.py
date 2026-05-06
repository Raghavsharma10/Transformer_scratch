def appendleft(self, value):
        """Add *value* to the left side of the collection."""
        def appendleft_trans(pipe):
            self._appendleft_helper(value, pipe)

        self._transaction(appendleft_trans)