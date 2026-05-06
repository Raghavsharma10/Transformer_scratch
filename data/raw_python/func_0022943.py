def _compute_bounds(self, axis, view):
        """Get the bounds

        Parameters
        ----------
        mode : str
            Describes the type of boundary requested. Can be "visual", "data",
            or "mouse".
        axis : 0, 1, 2
            The axis along which to measure the bounding values, in
            x-y-z order.
        """
        # Can and should we calculate bounds?
        if (self._bounds is None) and self._pos is not None:
            pos = self._pos
            self._bounds = [(pos[:, d].min(), pos[:, d].max())
                            for d in range(pos.shape[1])]
        # Return what we can
        if self._bounds is None:
            return
        else:
            if axis < len(self._bounds):
                return self._bounds[axis]
            else:
                return (0, 0)