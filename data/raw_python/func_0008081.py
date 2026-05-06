def _handle_outliers(self, p_o):
        """ Sets observation probabilities of outliers to uniform if ignore_outliers is set.
        Parameters
        ----------
        p_o : ndarray((T, N))
            output probabilities
        """
        if self.ignore_outliers:
            outliers = np.where(p_o.sum(axis=1)==0)[0]
            if outliers.size > 0:
                p_o[outliers, :] = 1.0
                self.found_outliers = True
        return p_o