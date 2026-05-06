def compute_joint(self):
        '''
        Compute the model in a single step, allowing for a light curve-wide
        transit model. This is a bit more expensive to compute.

        '''

        # Init
        log.info('Computing the joint model...')
        A = [None for b in self.breakpoints]
        B = [None for b in self.breakpoints]

        # We need to make sure that we're not masking the transits we are
        # trying to fit!
        # NOTE: If there happens to be an index that *SHOULD* be masked during
        # a transit (cosmic ray, detector anomaly), update `self.badmask`
        # to include that index.
        # Bad data points are *never* used in the regression.
        if self.transit_model is not None:
            outmask = np.array(self.outmask)
            transitmask = np.array(self.transitmask)
            transit_inds = np.where(
                np.sum([tm(self.time) for tm in self.transit_model],
                       axis=0) < 0)[0]
            self.outmask = np.array(
                [i for i in self.outmask if i not in transit_inds])
            self.transitmask = np.array(
                [i for i in self.transitmask if i not in transit_inds])

        # Loop over all chunks
        for b, brkpt in enumerate(self.breakpoints):

            # Masks for current chunk
            m = self.get_masked_chunk(b, pad=False)
            c = self.get_chunk(b, pad=False)

            # The X^2 matrices
            A[b] = np.zeros((len(m), len(m)))
            B[b] = np.zeros((len(c), len(m)))

            # Loop over all orders
            for n in range(self.pld_order):

                # Only compute up to the current PLD order
                if (self.lam_idx >= n) and (self.lam[b][n] is not None):
                    XM = self.X(n, m)
                    XC = self.X(n, c)
                    A[b] += self.lam[b][n] * np.dot(XM, XM.T)
                    B[b] += self.lam[b][n] * np.dot(XC, XM.T)
                    del XM, XC

        # Merge chunks. BIGA and BIGB are sparse, but unfortunately
        # scipy.sparse doesn't handle sparse matrix inversion all that
        # well when the *result* is not itself sparse. So we're sticking
        # with regular np.linalg.
        BIGA = block_diag(*A)
        del A
        BIGB = block_diag(*B)
        del B

        # Compute the full covariance matrix
        mK = GetCovariance(self.kernel, self.kernel_params, self.apply_mask(
             self.time), self.apply_mask(self.fraw_err))

        # The normalized, masked flux array
        f = self.apply_mask(self.fraw)
        med = np.nanmedian(f)
        f -= med

        # Are we computing a joint transit model?
        if self.transit_model is not None:

            # Get the unmasked indices
            m = self.apply_mask()

            # Subtract off the mean total transit model
            mean_transit_model = med * \
                np.sum([tm.depth * tm(self.time[m])
                        for tm in self.transit_model], axis=0)
            f -= mean_transit_model

            # Now add each transit model to the matrix of regressors
            for tm in self.transit_model:
                XM = tm(self.time[m]).reshape(-1, 1)
                XC = tm(self.time).reshape(-1, 1)
                BIGA += med ** 2 * tm.var_depth * np.dot(XM, XM.T)
                BIGB += med ** 2 * tm.var_depth * np.dot(XC, XM.T)
                del XM, XC

            # Dot the inverse of the covariance matrix
            W = np.linalg.solve(mK + BIGA, f)
            self.model = np.dot(BIGB, W)

            # Compute the transit weights and maximum likelihood transit model
            w_trn = med ** 2 * np.concatenate([tm.var_depth * np.dot(
                    tm(self.time[m]).reshape(1, -1), W)
                    for tm in self.transit_model])
            self.transit_depth = np.array(
                 [med * tm.depth + w_trn[i] for i, tm in
                  enumerate(self.transit_model)]) / med

            # Remove the transit prediction from the model
            self.model -= np.dot(np.hstack([tm(self.time).reshape(-1, 1)
                                 for tm in self.transit_model]),
                                 w_trn)

        else:

            # No transit model to worry about
            W = np.linalg.solve(mK + BIGA, f)
            self.model = np.dot(BIGB, W)

        # Subtract the global median
        self.model -= np.nanmedian(self.model)

        # Restore the mask
        if self.transit_model is not None:
            self.outmask = outmask
            self.transitmask = transitmask

        # Get the CDPP and reset the weights
        self.cdpp_arr = self.get_cdpp_arr()
        self.cdpp = self.get_cdpp()
        self._weights = None