def _updateTransitionMatrix(self):
        """
        Updates the hidden-state transition matrix and the initial distribution

        """
        # TRANSITION MATRIX
        C = self.model.count_matrix() + self.prior_C  # posterior count matrix

        # check if we work with these options
        if self.reversible and not _tmatrix_disconnected.is_connected(C, strong=True):
            raise NotImplementedError('Encountered disconnected count matrix with sampling option reversible:\n '
                                      + str(C) + '\nUse prior to ensure connectivity or use reversible=False.')
        # ensure consistent sparsity pattern (P0 might have additional zeros because of underflows)
        # TODO: these steps work around a bug in msmtools. Should be fixed there
        P0 = msmest.transition_matrix(C, reversible=self.reversible, maxiter=10000, warn_not_converged=False)
        zeros = np.where(P0 + P0.T == 0)
        C[zeros] = 0
        # run sampler
        Tij = msmest.sample_tmatrix(C, nsample=1, nsteps=self.transition_matrix_sampling_steps,
                                    reversible=self.reversible)

        # INITIAL DISTRIBUTION
        if self.stationary:  # p0 is consistent with P
            p0 = _tmatrix_disconnected.stationary_distribution(Tij, C=C)
        else:
            n0 = self.model.count_init().astype(float)
            first_timestep_counts_with_prior = n0 + self.prior_n0
            positive = first_timestep_counts_with_prior > 0
            p0 = np.zeros_like(n0)
            p0[positive] = np.random.dirichlet(first_timestep_counts_with_prior[positive])  # sample p0 from posterior

        # update HMM with new sample
        self.model.update(p0, Tij)