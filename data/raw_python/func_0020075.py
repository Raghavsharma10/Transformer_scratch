def lnlike(self, model, refactor=False, pos_tol=2.5, neg_tol=50.,
               full_output=False):
        r"""
        Return the likelihood of the astrophysical model `model`.

        Returns the likelihood of `model` marginalized over the PLD model.

        :param ndarray model: A vector of the same shape as `self.time` \
               corresponding to the astrophysical model.
        :param bool refactor: Re-compute the Cholesky decomposition? This \
               typically does not need to be done, except when the PLD \
               model changes. Default :py:obj:`False`.
        :param float pos_tol: the positive (i.e., above the median) \
               outlier tolerance in standard deviations.
        :param float neg_tol: the negative (i.e., below the median) \
               outlier tolerance in standard deviations.
        :param bool full_output: If :py:obj:`True`, returns the maximum \
               likelihood model amplitude and the variance on the amplitude \
               in addition to the log-likelihood. In the case of a transit \
               model, these are the transit depth and depth variance. Default \
               :py:obj:`False`.
        """
        lnl = 0

        # Re-factorize the Cholesky decomposition?
        try:
            self._ll_info
        except AttributeError:
            refactor = True
        if refactor:

            # Smooth the light curve and reset the outlier mask
            t = np.delete(self.time,
                          np.concatenate([self.nanmask, self.badmask]))
            f = np.delete(self.flux,
                          np.concatenate([self.nanmask, self.badmask]))
            f = SavGol(f)
            med = np.nanmedian(f)
            MAD = 1.4826 * np.nanmedian(np.abs(f - med))
            pos_inds = np.where((f > med + pos_tol * MAD))[0]
            pos_inds = np.array([np.argmax(self.time == t[i])
                                 for i in pos_inds])
            MAD = 1.4826 * np.nanmedian(np.abs(f - med))
            neg_inds = np.where((f < med - neg_tol * MAD))[0]
            neg_inds = np.array([np.argmax(self.time == t[i])
                                 for i in neg_inds])
            outmask = np.array(self.outmask)
            transitmask = np.array(self.transitmask)
            self.outmask = np.concatenate([neg_inds, pos_inds])
            self.transitmask = np.array([], dtype=int)

            # Now re-factorize the Cholesky decomposition
            self._ll_info = [None for b in self.breakpoints]
            for b, brkpt in enumerate(self.breakpoints):

                # Masks for current chunk
                m = self.get_masked_chunk(b, pad=False)

                # This block of the masked covariance matrix
                K = GetCovariance(self.kernel, self.kernel_params,
                                  self.time[m], self.fraw_err[m])

                # The masked X.L.X^T term
                A = np.zeros((len(m), len(m)))
                for n in range(self.pld_order):
                    XM = self.X(n, m)
                    A += self.lam[b][n] * np.dot(XM, XM.T)
                K += A
                self._ll_info[b] = [cho_factor(K), m]

            # Reset the outlier masks
            self.outmask = outmask
            self.transitmask = transitmask

        # Compute the likelihood for each chunk
        amp = [None for b in self.breakpoints]
        var = [None for b in self.breakpoints]
        for b, brkpt in enumerate(self.breakpoints):
            # Get the inverse covariance and the mask
            CDK = self._ll_info[b][0]
            m = self._ll_info[b][1]
            # Compute the maximum likelihood model amplitude
            # (for transits, this is the transit depth)
            var[b] = 1. / np.dot(model[m], cho_solve(CDK, model[m]))
            amp[b] = var[b] * np.dot(model[m], cho_solve(CDK, self.fraw[m]))
            # Compute the residual
            r = self.fraw[m] - amp[b] * model[m]
            # Finally, compute the likelihood
            lnl += -0.5 * np.dot(r, cho_solve(CDK, r))

        if full_output:
            # We need to multiply the Gaussians for all chunks to get the
            # amplitude and amplitude variance for the entire dataset
            vari = var[0]
            ampi = amp[0]
            for v, a in zip(var[1:], amp[1:]):
                ampi = (ampi * v + a * vari) / (vari + v)
                vari = vari * v / (vari + v)
            med = np.nanmedian(self.fraw)
            return lnl, ampi / med, vari / med ** 2
        else:
            return lnl