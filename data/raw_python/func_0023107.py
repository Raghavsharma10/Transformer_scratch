def bounds(self, axis, view=None):
        """Get the bounds of the Visual

        Parameters
        ----------
        axis : int
            The axis.
        view : instance of VisualView
            The view to use.
        """
        if view is None:
            view = self
        if axis not in self._vshare.bounds:
            self._vshare.bounds[axis] = self._compute_bounds(axis, view)
        return self._vshare.bounds[axis]