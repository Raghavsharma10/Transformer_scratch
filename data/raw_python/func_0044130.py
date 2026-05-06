def contains(self, order, cell, include_smaller=False):
        """Test whether the MOC contains the given cell.

        If the include_smaller argument is true then the MOC is considered
        to include a cell if it includes part of that cell (at a higher
        order).

        >>> m = MOC(1, (5,))
        >>> m.contains(0, 0)
        False
        >>> m.contains(0, 1, True)
        True
        >>> m.contains(0, 1, False)
        False
        >>> m.contains(1, 4)
        False
        >>> m.contains(1, 5)
        True
        >>> m.contains(2, 19)
        False
        >>> m.contains(2, 21)
        True
        """

        order = self._validate_order(order)
        cell = self._validate_cell(order, cell)

        return self._compare_operation(order, cell, include_smaller, 'check')