def space(self, newlines=1):
        """Creates a vertical space of newlines

        Args:
            newlines (int): number of empty lines

        Returns:
            self for chaining
        """
        space = Space()
        for line in range(newlines):
            space.add_line('\n')
        self._container.structure.insert(self._idx, space)
        self._idx += 1
        return self