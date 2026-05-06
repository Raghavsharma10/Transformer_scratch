def compute(self):
        '''
        Compute the model for the current value of lambda.

        '''

        # Is there a transit model?
        if self.transit_model is not None:
            return self.compute_joint()

        log.info('Computing the model...')

        # Loop over all chunks
        model = [None for b in self.breakpoints]
        for b, brkpt in enumerate(self.breakpoints):

            # Masks for current chunk
            m = self.get_masked_chunk(b)
            c = self.get_chunk(b)

            # This block of the masked covariance matrix
            mK = GetCovariance(self.kernel, self.kernel_params,
                               self.time[m], self.fraw_err[m])

            # Get median
            med = np.nanmedian(self.fraw[m])

            # Normalize the flux
            f = self.fraw[m] - med

            # The X^2 matrices
            A = np.zeros((len(m), len(m)))
            B = np.zeros((len(c), len(m)))

            # Loop over all orders
            for n in range(self.pld_order):

                # Only compute up to the current PLD order
                if (self.lam_idx >= n) and (self.lam[b][n] is not None):
                    XM = self.X(n, m)
                    XC = self.X(n, c)
                    A += self.lam[b][n] * np.dot(XM, XM.T)
                    B += self.lam[b][n] * np.dot(XC, XM.T)
                    del XM, XC

            # Compute the model
            W = np.linalg.solve(mK + A, f)
            model[b] = np.dot(B, W)

        # Free up some memory
        del A, B, W

        # Join the chunks after applying the correct offset
        if len(model) > 1:

            # First chunk
            self.model = model[0][:-self.bpad]

            # Center chunks
            for m in model[1:-1]:
                # Join the chunks at the first non-outlier cadence
                i = 1
                while len(self.model) - i in self.mask:
                    i += 1
                offset = self.model[-i] - m[self.bpad - i]
                self.model = np.concatenate(
                    [self.model, m[self.bpad:-self.bpad] + offset])

            # Last chunk
            i = 1
            while len(self.model) - i in self.mask:
                i += 1
            offset = self.model[-i] - model[-1][self.bpad - i]
            self.model = np.concatenate(
                [self.model, model[-1][self.bpad:] + offset])

        else:

            self.model = model[0]

        # Subtract the global median
        self.model -= np.nanmedian(self.model)

        # Get the CDPP and reset the weights
        self.cdpp_arr = self.get_cdpp_arr()
        self.cdpp = self.get_cdpp()
        self._weights = None