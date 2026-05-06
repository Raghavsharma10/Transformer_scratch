def _apply_axes_mapping(self, target, inverse=False):
        """
        Apply the transposition to the target iterable.

        Parameters
        ----------
        target - iterable
            The iterable to transpose. This would be suitable for things
            such as a shape as well as a list of ``__getitem__`` keys.
        inverse - bool
            Whether to map old dimension to new dimension (forward), or
            new dimension to old dimension (inverse). Default is False
            (forward).

        Returns
        -------
        A tuple derived from target which has been ordered based on the new
        axes.

        """
        if len(target) != self.ndim:
            raise ValueError('The target iterable is of length {}, but '
                             'should be of length {}.'.format(len(target),
                                                              self.ndim))
        if inverse:
            axis_map = self._inverse_axes_map
        else:
            axis_map = self._forward_axes_map

        result = [None] * self.ndim
        for axis, item in enumerate(target):
            result[axis_map[axis]] = item
        return tuple(result)