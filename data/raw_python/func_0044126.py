def add(self, order, cells, no_validation=False):
        """Add cells at a given order to the MOC.

        The cells are inserted into the MOC at the specified order.  This
        leaves the MOC in an un-normalized state.  The cells are given
        as a collection of integers (or types which can be converted
        to integers).

        >>> m = MOC()
        >>> m.add(4, (20, 21))
        >>> m.cells
        2

        >>> m.add(5, (88, 89))
        >>> m.cells
        4

        The `no_validation` option can be given to skip validation of the
        cell numbers.  They must already be integers in the correct range.
        """

        self._normalized = False

        order = self._validate_order(order)

        if no_validation:
            # Simply add the given cells to the set with no validation.
            self._orders[order].update(cells)

        else:
            # Collect validated cell numbers in a set for addition.
            cell_set = set()

            for cell in cells:
                cell = self._validate_cell(order, cell)
                cell_set.add(cell)

            self._orders[order].update(cell_set)