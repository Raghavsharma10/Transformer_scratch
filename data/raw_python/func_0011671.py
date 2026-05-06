def reshape(self, newshape, order='C'):
        """If axis 0 is unaffected by the reshape, then returns a Timeseries,
        otherwise returns an ndarray. Preserves labels of axis j only if all 
        axes<=j are unaffected by the reshape.  
        See ``numpy.ndarray.reshape()`` for more information
        """
        oldshape = self.shape
        ar = np.asarray(self).reshape(newshape, order=order)
        if (newshape is -1 and len(oldshape) is 1 or
                (isinstance(newshape, numbers.Integral) and 
                    newshape == oldshape[0]) or 
                (isinstance(newshape, Sequence) and
                    (newshape[0] == oldshape[0] or
                     (newshape[0] is -1 and np.array(oldshape[1:]).prod() ==
                                            np.array(newshape[1:]).prod())))):
            # then axis 0 is unaffected by the reshape
            newlabels = [None] * ar.ndim
            i = 1
            while i < ar.ndim and i < self.ndim and ar.shape[i] == oldshape[i]:
                newlabels[i] = self.labels[i]
                i += 1
            return Timeseries(ar, self.tspan, newlabels)
        else:
            return ar