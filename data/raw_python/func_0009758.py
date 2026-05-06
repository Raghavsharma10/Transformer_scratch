def iter_starts_with_index(self):
        """Returns a generator over (index, start coordinate) pairs,
        excluding the origin."""
        for i in range(1, len(self.endpoints)):
            yield i, self.get_coordinates(i)