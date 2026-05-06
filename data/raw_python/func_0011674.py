def concatenate(self, tup, axis=0):
        """Join a sequence of Timeseries to this one
        Args: 
          tup (sequence of Timeseries): timeseries to be joined with this one.
            They must have the same shape as this Timeseries, except in the
            dimension corresponding to `axis`.
          axis (int, optional): The axis along which timeseries will be joined.
        Returns:
          res (Timeseries or ndarray)
        """
        if not isinstance(tup, Sequence):
            tup = (tup,)
        if tup is (None,) or len(tup) is 0:
            return self
        tup = (self,) + tuple(tup)
        new_array = np.concatenate(tup, axis)
        if not all(hasattr(ts, 'tspan') and 
                   hasattr(ts, 'labels') for ts in tup):
            return new_array
        if axis == 0:
            starts = [ts.tspan[0] for ts in tup]
            ends = [ts.tspan[-1] for ts in tup]
            if not all(starts[i] > ends[i-1] for i in range(1, len(starts))):
                # series being joined are not ordered in time. not Timeseries
                return new_array
            else:
                new_tspan = np.concatenate([ts.tspan for ts in tup])
        else:
            new_tspan = self.tspan
        new_labels = [None]
        for ax in range(1, new_array.ndim):
            if ax == axis:
                axislabels = []
                for ts in tup:
                    if ts.labels[axis] is None:
                        axislabels.extend('' * ts.shape[axis])
                    else:
                        axislabels.extend(ts.labels[axis])
                if all(lab == '' for lab in axislabels):
                    new_labels.append(None)
                else:
                    new_labels.append(axislabels)
            else:
                # non-concatenation axis
                axlabels = tup[0].labels[ax]
                if not all(ts.labels[ax] == axlabels for ts in tup[1:]):
                    # series to be joined do not agree on labels for this axis
                    axlabels = None
                new_labels.append(axlabels)
        return self.__new__(self.__class__, new_array, new_tspan, new_labels)