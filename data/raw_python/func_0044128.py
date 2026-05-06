def clear(self):
        """Clears all cells from a MOC.

        >>> m = MOC(4, (5, 6))
        >>> m.clear()
        >>> m.cells
        0
        """

        for order in range(0, MAX_ORDER + 1):
            self._orders[order].clear()

        self._normalized = True