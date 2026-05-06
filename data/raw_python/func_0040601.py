def iter_members(self):
        """
        Iterate over all elements of the set.

        :raises ValueError: if self is a set of everything
        """
        if self.everything:
            raise ValueError("Can not iterate everything")
        for coord in sorted(self.elements):
            yield coord