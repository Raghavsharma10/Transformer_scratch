def swapaxes(self, axis1, axis2):
        """Interchange two axes of a Timeseries."""
        if self.ndim <=1 or axis1 == axis2:
            return self
        ar = np.asarray(self).swapaxes(axis1, axis2)
        if axis1 != 0 and axis2 != 0:
            # then axis 0 is unaffected by the swap
            labels = self.labels[:]
            labels[axis1], labels[axis2] = labels[axis2], labels[axis1]
            return Timeseries(ar, self.tspan, labels)
        return ar