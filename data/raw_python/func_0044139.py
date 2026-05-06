def _validate_cell(self, order, cell):
        """Check that the given cell is valid.

        The order is assumed already to have been validated.
        """

        max_cells = self._order_num_cells(order)

        try:
            cell = int(cell)
        except ValueError as e:
            raise TypeError('MOC cell must be convertable to int')

        if not 0 <= cell < max_cells:
            raise ValueError(
                'MOC cell order {0} must be in range 0-{1}'.format(
                    order, max_cells - 1))

        return cell