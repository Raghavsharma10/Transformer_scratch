def apply_mask(self, x=None):
        '''
        Returns the outlier mask, an array of indices corresponding to the
        non-outliers.

        :param numpy.ndarray x: If specified, returns the masked version of \
               :py:obj:`x` instead. Default :py:obj:`None`

        '''

        if x is None:
            return np.delete(np.arange(len(self.time)), self.mask)
        else:
            return np.delete(x, self.mask, axis=0)