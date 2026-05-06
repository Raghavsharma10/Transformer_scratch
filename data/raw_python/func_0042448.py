def items(self):
        """
        ITERATE THROUGH ALL coord, value PAIRS
        """
        for c in self._all_combos():
            _, value = _getitem(self.cube, c)
            yield c, value