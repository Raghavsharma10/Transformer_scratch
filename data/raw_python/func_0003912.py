def _skip_frame(self):
        """Skip the next time frame"""
        for line in self._f:
            if line == 'ITEM: ATOMS\n':
                break
        for i in range(self.num_atoms):
            next(self._f)