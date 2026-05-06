def _get_norm(self):
        '''
        Computes the PLD flux normalization array.

        ..note :: `iPLD` model **only**.

        '''

        log.info('Computing the PLD normalization...')

        # Loop over all chunks
        mod = [None for b in self.breakpoints]
        for b, brkpt in enumerate(self.breakpoints):

            # Unmasked chunk
            c = self.get_chunk(b)

            # Masked chunk (original mask plus user transit mask)
            inds = np.array(
                list(set(np.concatenate([self.transitmask,
                                         self.recmask]))), dtype=int)
            M = np.delete(np.arange(len(self.time)), inds, axis=0)
            if b > 0:
                m = M[(M > self.breakpoints[b - 1] - self.bpad)
                      & (M <= self.breakpoints[b] + self.bpad)]
            else:
                m = M[M <= self.breakpoints[b] + self.bpad]

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
                XM = self.X(n, m)
                XC = self.X(n, c)
                A += self.reclam[b][n] * np.dot(XM, XM.T)
                B += self.reclam[b][n] * np.dot(XC, XM.T)
                del XM, XC

            W = np.linalg.solve(mK + A, f)
            mod[b] = np.dot(B, W)
            del A, B, W

        # Join the chunks after applying the correct offset
        if len(mod) > 1:

            # First chunk
            model = mod[0][:-self.bpad]

            # Center chunks
            for m in mod[1:-1]:
                offset = model[-1] - m[self.bpad - 1]
                model = np.concatenate(
                    [model, m[self.bpad:-self.bpad] + offset])

            # Last chunk
            offset = model[-1] - mod[-1][self.bpad - 1]
            model = np.concatenate([model, mod[-1][self.bpad:] + offset])

        else:

            model = mod[0]

        # Subtract the global median
        model -= np.nanmedian(model)

        # Save the norm
        self._norm = self.fraw - model