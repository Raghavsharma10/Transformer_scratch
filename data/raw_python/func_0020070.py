def get_chunk(self, b, x=None, pad=True):
        '''
        Returns the indices corresponding to a given light curve chunk.

        :param int b: The index of the chunk to return
        :param numpy.ndarray x: If specified, applies the mask to array \
               :py:obj:`x`. Default :py:obj:`None`

        '''

        M = np.arange(len(self.time))
        if b > 0:
            res = M[(M > self.breakpoints[b - 1] - int(pad) * self.bpad)
                    & (M <= self.breakpoints[b] + int(pad) * self.bpad)]
        else:
            res = M[M <= self.breakpoints[b] + int(pad) * self.bpad]
        if x is None:
            return res
        else:
            return x[res]