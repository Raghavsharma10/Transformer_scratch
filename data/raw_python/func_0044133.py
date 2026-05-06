def normalize(self, max_order=MAX_ORDER):
        """Ensure that the MOC is "well-formed".

        This structures the MOC as is required for the FITS and JSON
        representation.  This method is invoked automatically when writing
        to these formats.

        The number of cells in the MOC will be minimized, so that
        no area of the sky is covered multiple times by cells at
        different orders, and if all four neighboring cells are
        present at an order (other than order 0), they are merged
        into their parent cell at the next lower order.

        >>> m = MOC(1, (0, 1, 2, 3))
        >>> m.cells
        4

        >>> m.normalize()
        >>> m.cells
        1
        """

        max_order = self._validate_order(max_order)

        # If the MOC is already normalized and we are not being asked
        # to reduce the order, then do nothing.
        if self.normalized and max_order >= self.order:
            return

        # Group the pixels by iterating down from the order.  At each
        # order, where all 4 adjacent pixels are present (or we are above
        # the maximum order) they are replaced with a single pixel in the
        # next lower order.  Otherwise the pixel should appear in the MOC
        # unless it is already represented at a lower order.
        for order in range(self.order, 0, -1):
            pixels = self._orders[order]

            next_pixels = self._orders[order - 1]

            new_pixels = set()

            while pixels:
                pixel = pixels.pop()

                # Look to lower orders to ensure this pixel isn't
                # already covered.
                check_pixel = pixel
                already_contained = True
                for check_order in range(order - 1, -1, -1):
                    check_pixel >>= 2
                    if check_pixel in self._orders[check_order]:
                        break
                else:
                    already_contained = False

                # Check whether this order is above the maximum, or
                # if we have all 4 adjacent pixels.  Also do this if
                # the pixel was already contained at a lower level
                # so that we can avoid checking the adjacent pixels.
                if (already_contained or (order > max_order) or
                        (((pixel ^ 1) in pixels) and
                         ((pixel ^ 2) in pixels) and
                         ((pixel ^ 3) in pixels))):

                    pixels.discard(pixel ^ 1)
                    pixels.discard(pixel ^ 2)
                    pixels.discard(pixel ^ 3)

                    if not already_contained:
                        # Group these pixels by placing the equivalent pixel
                        # for the next order down in the set.
                        next_pixels.add(pixel >> 2)

                else:
                    new_pixels.add(pixel)

            if new_pixels:
                self._orders[order].update(new_pixels)

        self._normalized = True