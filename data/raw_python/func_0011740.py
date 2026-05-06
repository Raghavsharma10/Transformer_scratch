def expand_dims(self, axis):
        """Insert a new axis, at a given position in the array shape
        Args:
          axis (int): Position (amongst axes) where new axis is to be inserted.
        """
        if axis <= self._distaxis:
            subaxis = axis
            new_distaxis = self._distaxis + 1
        else:
            subaxis = axis - 1
            new_distaxis = self._distaxis
        new_subts = [rts.expand_dims(subaxis) for rts in self._subarrays]
        if axis == 0:
            # prepended an axis: no longer a Timeseries
            return distob.DistArray(new_subts, new_distaxis)
        else:
            axislabels = self.labels[self._distaxis]
            return DistTimeseries(new_subts, new_distaxis, axislabels)