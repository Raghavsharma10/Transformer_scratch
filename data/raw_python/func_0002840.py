def insert(self, key, value):
        '''Adds a new key-value pair. Returns any discarded values.'''

        # Add to history and catch expectorate
        if len(self.history) == self.maxsize:
            expectorate = self.history[0]
        else:
            expectorate = None

        self.history.append((key, value))

        # Add to the appropriate list of values
        if key in self:
            super().__getitem__(key).append(value)
        else:
            super().__setitem__(key, [value])

        # Clean up old values
        if expectorate is not None:
            old_key, old_value = expectorate
            super().__getitem__(old_key).pop(0)
            if len(super().__getitem__(old_key)) == 0:
                super().__delitem__(old_key)

            return (old_key, old_value)