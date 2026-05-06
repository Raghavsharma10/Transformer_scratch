def remove(self, order, cells):
        """Remove cells at a given order from the MOC.
        """

        self._normalized = False

        order = self._validate_order(order)

        for cell in cells:
            cell = self._validate_cell(order, cell)

            self._compare_operation(order, cell, True, 'remove')