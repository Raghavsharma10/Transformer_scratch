def _compute_bounds(self, axis, view):
        """Return the (min, max) bounding values of this visual along *axis*
        in the local coordinate system.
        """
        is_vertical = self._is_vertical
        pos = self._pos
        if axis == 0 and is_vertical:
            return (pos[0, 0], pos[0, 0])
        elif axis == 1 and not is_vertical:
            return (self._pos[0, 1], self._pos[0, 1])

        return None