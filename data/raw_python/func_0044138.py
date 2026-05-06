def _validate_order(self, order):
        """Check that the given order is valid."""

        try:
            order = int(order)
        except ValueError as e:
            raise TypeError('MOC order must be convertable to int')

        if not 0 <= order <= MAX_ORDER:
            raise ValueError(
                'MOC order must be in range 0-{0}'.format(MAX_ORDER))

        return order