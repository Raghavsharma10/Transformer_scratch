def random_draw(self, size=None):
        """Draw random samples of the hyperparameters.
        
        Parameters
        ----------
        size : None, int or array-like, optional
            The number/shape of samples to draw. If None, only one sample is
            returned. Default is None.
        """
        if size is None:
            size = 1
            single_val = True
        else:
            single_val = False
        
        out_shape = [len(self.bounds)]
        try:
            out_shape.extend(size)
        except TypeError:
            out_shape.append(size)
        
        out = scipy.zeros(out_shape)
        for j in xrange(0, len(self.bounds)):
            if j != 2:
                out[j, :] = numpy.random.uniform(low=self.bounds[j][0],
                                                 high=self.bounds[j][1],
                                                 size=size)
            else:
                out[j, :] = numpy.random.uniform(low=self.bounds[j][0],
                                                 high=out[j - 1, :],
                                                 size=size)
        if not single_val:
            return out
        else:
            return out.ravel()