def expand_dims(self, axis):
        """Insert a new axis, at a given position in the array shape
        Args:
          axis (int): Position (amongst axes) where new axis is to be inserted.
        """
        if axis == -1:
            axis = self.ndim
        array = np.expand_dims(self, axis)
        if axis == 0:
            # prepended an axis: no longer a Timeseries
            return array
        else:
            new_labels = self.labels.insert(axis, None)
            return Timeseries(array, self.tspan, new_labels)