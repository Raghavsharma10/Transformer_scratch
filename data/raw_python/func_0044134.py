def flattened(self, order=None, include_smaller=True):
        """Return a flattened pixel collection at a single order."""

        if order is None:
            order = self.order
        else:
            order = self._validate_order(order)

        # Start with the cells which are already at this order.
        flat = set(self[order])

        # Look at lower orders and expand them into this set.
        # Based on the "map" algorithm from Appendix A of the
        # MOC recommendation.
        for order_i in range(0, order):
            shift = 2 * (order - order_i)

            for cell in self[order_i]:
                flat.update(range(cell << shift, (cell + 1) << shift))

        # Look at higher orders unless we have been told to exclude
        # them.
        if include_smaller:
            for order_i in range(order + 1, MAX_ORDER + 1):
                shift = 2 * (order_i - order)

                for cell in self[order_i]:
                    flat.add(cell >> shift)

        return flat