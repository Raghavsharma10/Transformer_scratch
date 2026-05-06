def get_weights(self):
        '''
        Computes the PLD weights vector :py:obj:`w`.

        ..warning :: Deprecated and not thoroughly tested.

        '''

        log.info("Computing PLD weights...")

        # Loop over all chunks
        weights = [None for i in range(len(self.breakpoints))]
        for b, brkpt in enumerate(self.breakpoints):

            # Masks for current chunk
            m = self.get_masked_chunk(b)
            c = self.get_chunk(b)

            # This block of the masked covariance matrix
            _mK = GetCovariance(self.kernel, self.kernel_params,
                                self.time[m], self.fraw_err[m])

            # This chunk of the normalized flux
            f = self.fraw[m] - np.nanmedian(self.fraw)

            # Loop over all orders
            _A = [None for i in range(self.pld_order)]
            for n in range(self.pld_order):
                if self.lam_idx >= n:
                    X = self.X(n, m)
                    _A[n] = np.dot(X, X.T)
                    del X

            # Compute the weights
            A = np.sum([l * a for l, a in zip(self.lam[b], _A)
                        if l is not None], axis=0)
            W = np.linalg.solve(_mK + A, f)
            weights[b] = [l * np.dot(self.X(n, m).T, W)
                          for n, l in enumerate(self.lam[b]) if l is not None]

        self._weights = weights