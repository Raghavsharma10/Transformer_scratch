def _compare_operation(self, order, cell, include_smaller, operation):
        """General internal method for comparison-based operations.

        This is a private method, and does not update the normalized
        flag.
        """

        # Check for a larger cell (lower order) which contains the
        # given cell.
        for order_i in range(0, order):
            shift = 2 * (order - order_i)
            cell_i = cell >> shift

            if cell_i in self._orders[order_i]:
                if operation == 'check':
                    return True
                elif operation == 'remove':
                    # Remove the cell and break it into its 4 constituent
                    # cells.  Those which actually match the area we are
                    # trying to remove will be removed at the next stage.
                    self._orders[order_i].remove(cell_i)
                    self.add(order_i + 1,
                             range(cell_i << 2, (cell_i + 1) << 2))
                elif operation == 'inter':
                    return [(order, (cell,))]

        # Check for the specific cell itself, but only after looking at larger
        # cells because for the "remove" operation we may have broken up
        # one of the large cells so that it subsequently matches.
        if cell in self._orders[order]:
            if operation == 'check':
                return True
            elif operation == 'remove':
                self._orders[order].remove(cell)
            elif operation == 'inter':
                return [(order, (cell,))]

        result = []

        if include_smaller:
            # Check for a smaller cell (higher order) which is part
            # of the given cell.
            for order_i in range(order + 1, MAX_ORDER + 1):
                shift = 2 * (order_i - order)

                cells = []

                for cell_i in self._orders[order_i]:
                    if (cell_i >> shift) == cell:
                        if operation == 'check':
                            return True
                        elif operation == 'remove' or operation == 'inter':
                            cells.append(cell_i)

                if operation == 'remove':
                    for cell_i in cells:
                        self._orders[order_i].remove(cell_i)
                elif operation == 'inter':
                    if cells:
                        result.append((order_i, cells))

        if operation == 'check':
            return False
        elif operation == 'inter':
            return result