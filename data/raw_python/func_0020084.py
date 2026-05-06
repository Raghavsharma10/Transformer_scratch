def cv_precompute(self, mask, b):
        '''
        Pre-compute the matrices :py:obj:`A` and :py:obj:`B`
        (cross-validation step only)
        for chunk :py:obj:`b`.

        '''

        # Get current chunk and mask outliers
        m1 = self.get_masked_chunk(b)
        flux = self.fraw[m1]
        K = GetCovariance(self.kernel, self.kernel_params,
                          self.time[m1], self.fraw_err[m1])
        med = np.nanmedian(flux)

        # Now mask the validation set
        M = lambda x, axis = 0: np.delete(x, mask, axis=axis)
        m2 = M(m1)
        mK = M(M(K, axis=0), axis=1)
        f = M(flux) - med

        # Pre-compute the matrices
        A = [None for i in range(self.pld_order)]
        B = [None for i in range(self.pld_order)]
        for n in range(self.pld_order):
            # Only compute up to the current PLD order
            if self.lam_idx >= n:
                X2 = self.X(n, m2)
                X1 = self.X(n, m1)
                A[n] = np.dot(X2, X2.T)
                B[n] = np.dot(X1, X2.T)
                del X1, X2

        if self.transit_model is None:
            C = 0
        else:
            C = np.zeros((len(m2), len(m2)))
            mean_transit_model = med * \
                np.sum([tm.depth * tm(self.time[m2])
                        for tm in self.transit_model], axis=0)
            f -= mean_transit_model
            for tm in self.transit_model:
                X2 = tm(self.time[m2]).reshape(-1, 1)
                C += tm.var_depth * np.dot(X2, X2.T)
                del X2

        return A, B, C, mK, f, m1, m2