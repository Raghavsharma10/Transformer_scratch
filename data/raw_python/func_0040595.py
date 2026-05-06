def iter_members(self):
        """
        Iterate over all elements of the set.

        :raises ValueError: if self is a set of everything
        """
        if not self.is_discrete():
            raise ValueError("non-discrete IntervalSet can not be iterated")
        for i in self.ints:
            yield i.get_point()