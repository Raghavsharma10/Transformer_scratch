def optimize_lambda(self, validation):
        '''
        Returns the index of :py:attr:`self.lambda_arr` that minimizes the
        validation scatter in the segment with minimum at the lowest value
        of :py:obj:`lambda`, with
        fractional tolerance :py:attr:`self.leps`.

        :param numpy.ndarray validation: The scatter in the validation set \
               as a function of :py:obj:`lambda`

        '''

        maxm = 0
        minr = len(validation)
        for n in range(validation.shape[1]):
            # The index that minimizes the scatter for this segment
            m = np.nanargmin(validation[:, n])
            if m > maxm:
                # The largest of the `m`s.
                maxm = m
            # The largest index with validation scatter within
            # `self.leps` of the minimum for this segment
            r = np.where((validation[:, n] - validation[m, n]) /
                         validation[m, n] <= self.leps)[0][-1]
            if r < minr:
                # The smallest of the `r`s
                minr = r
        return min(maxm, minr)