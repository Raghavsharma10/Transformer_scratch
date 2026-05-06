def iter_surrounding(self, center_key):
        """Iterate over all bins surrounding the given bin"""
        for shift in self.neighbor_indexes:
            key = tuple(np.add(center_key, shift).astype(int))
            if self.integer_cell is not None:
                key = self.wrap_key(key)
            bin = self._bins.get(key)
            if bin is not None:
                yield key, bin