def transpose(self, *axes):
        """Permute the dimensions of a Timeseries."""
        if self.ndim <= 1:
            return self
        ar = np.asarray(self).transpose(*axes)
        if axes[0] != 0:
            # then axis 0 is unaffected by the transposition
            newlabels = [self.labels[ax] for ax in axes]
            return Timeseries(ar, self.tspan, newlabels)
        else:
            return ar