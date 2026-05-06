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
        
        out_shape = [self.num_var]
        try:
            out_shape.extend(size)
        except TypeError:
            out_shape.append(size)
        
        out = scipy.sort(
            numpy.random.uniform(
                low=self.lb,
                high=self.ub,
                size=out_shape
            ),
            axis=0
        )
        if not single_val:
            return out
        else:
            return out.ravel()