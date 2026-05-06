def solution_path(self):
        '''Follows the solution path of the generalized lasso to find the best lambda value.'''
        lambda_grid = np.exp(np.linspace(np.log(self.max_lambda), np.log(self.min_lambda), self.lambda_bins))
        aic_trace = np.zeros((len(self.bins),lambda_grid.shape[0])) # The AIC score for each lambda value
        aicc_trace = np.zeros((len(self.bins),lambda_grid.shape[0])) # The AICc score for each lambda value (correcting for finite sample size)
        bic_trace = np.zeros((len(self.bins),lambda_grid.shape[0])) # The BIC score for each lambda value
        dof_trace = np.zeros((len(self.bins),lambda_grid.shape[0])) # The degrees of freedom of each final solution
        log_likelihood_trace = np.zeros((len(self.bins),lambda_grid.shape[0]))
        bic_best_idx = [None for _ in self.bins]
        aic_best_idx = [None for _ in self.bins]
        aicc_best_idx = [None for _ in self.bins]
        bic_best_betas = [None for _ in self.bins]
        aic_best_betas = [None for _ in self.bins]
        aicc_best_betas = [None for _ in self.bins]
        if self.k == 0 and self.trails is not None:
            betas = [np.zeros(self.num_nodes, dtype='double') for _ in self.bins]
            zs = [np.zeros(self.breakpoints[-1], dtype='double') for _ in self.bins]
            us = [np.zeros(self.breakpoints[-1], dtype='double') for _ in self.bins]
        else:
            betas = [np.zeros(self.num_nodes, dtype='double') for _ in self.bins]
            us = [np.zeros(self.Dk.shape[0], dtype='double') for _ in self.bins]
        for i, _lambda in enumerate(lambda_grid):
            if self.verbose:
                print('\n#{0} Lambda = {1}'.format(i, _lambda))

            # Run the graph fused lasso over each bin with the current lambda value
            initial_values = (betas, zs, us) if self.k == 0 and self.trails is not None else (betas, us)
            self.run(_lambda, initial_values=initial_values)

            if self.verbose > 1:
                print('\tCalculating degrees of freedom and information criteria')

            for b, beta in enumerate(betas):
                if self.bins_allowed is not None and b not in self.bins_allowed:
                    continue

                # Count the number of free parameters in the grid (dof)
                # TODO: this is not really the true DoF, since a change in a higher node multiplies
                # the DoF in the lower nodes
                # dof_trace[b,i] = len(self.calc_plateaus(beta))
                dof_vals = self.Dk_minus_one.dot(beta) if self.k > 0 else beta
                plateaus = calc_plateaus(dof_vals, self.edges, rel_tol=0.01) if (self.k % 2) == 0 else nearly_unique(dof_vals, rel_tol=0.03)
                #plateaus = calc_plateaus(dof_vals, self.edges, rel_tol=1e-5) if (self.k % 2) == 0 else nearly_unique(dof_vals, rel_tol=1e-5)
                dof_trace[b,i] = max(1,len(plateaus)) #* (k+1)

                # Get the negative log-likelihood
                log_likelihood_trace[b,i] = self.data_log_likelihood(self.bins[b][-1], self.bins[b][-2], beta)

                # Calculate AIC = 2k - 2ln(L)
                aic_trace[b,i] = 2. * dof_trace[b,i] - 2. * log_likelihood_trace[b,i]
                
                # Calculate AICc = AIC + 2k * (k+1) / (n - k - 1)
                aicc_trace[b,i] = aic_trace[b,i] + 2 * dof_trace[b,i] * (dof_trace[b,i]+1) / (self.num_nodes - dof_trace[b,i] - 1.)

                # Calculate BIC = -2ln(L) + k * (ln(n) - ln(2pi))
                bic_trace[b,i] = -2 * log_likelihood_trace[b,i] + dof_trace[b,i] * (np.log(self.num_nodes) - np.log(2 * np.pi))

                # Track the best model thus far
                if aic_best_idx[b] is None or aic_trace[b,i] < aic_trace[b,aic_best_idx[b]]:
                    aic_best_idx[b] = i
                    aic_best_betas[b] = np.array(beta)

                # Track the best model thus far
                if aicc_best_idx[b] is None or aicc_trace[b,i] < aicc_trace[b,aicc_best_idx[b]]:
                    aicc_best_idx[b] = i
                    aicc_best_betas[b] = np.array(beta)

                # Track the best model thus far
                if bic_best_idx[b] is None or bic_trace[b,i] < bic_trace[b,bic_best_idx[b]]:
                    bic_best_idx[b] = i
                    bic_best_betas[b] = np.array(beta)

                if self.verbose and self.bins_allowed is not None:
                    print('\tBin {0} Log-Likelihood: {1} DoF: {2} AIC: {3} AICc: {4} BIC: {5}'.format(b, log_likelihood_trace[b,i], dof_trace[b,i], aic_trace[b,i], aicc_trace[b,i], bic_trace[b,i]))

            if self.verbose and self.bins_allowed is None:
                print('Overall Log-Likelihood: {0} DoF: {1} AIC: {2} AICc: {3} BIC: {4}'.format(log_likelihood_trace[:,i].sum(), dof_trace[:,i].sum(), aic_trace[:,i].sum(), aicc_trace[:,i].sum(), bic_trace[:,i].sum()))

        if self.verbose:
            print('')
            print('Best settings per bin:')
            for b, (aic_idx, aicc_idx, bic_idx) in enumerate(zip(aic_best_idx, aicc_best_idx, bic_best_idx)):
                if self.bins_allowed is not None and b not in self.bins_allowed:
                    continue
                left, mid, right, trials, successes = self.bins[b]
                print('\tBin #{0} ([{1}, {2}], split={3}) lambda: AIC={4:.2f} AICC={5:.2f} BIC={6:.2f} DoF: AIC={7:.0f} AICC={8:.0f} BIC={9:.0f}'.format(
                        b, left, right, mid,
                        lambda_grid[aic_idx], lambda_grid[aicc_idx], lambda_grid[bic_idx],
                        dof_trace[b,aic_idx], dof_trace[b,aicc_idx], dof_trace[b,bic_idx]))
            print('')

        if self.bins_allowed is None:
            if self.verbose:
                print('Creating densities from betas...')
            bic_density = self.density_from_betas(bic_best_betas)
            aic_density = self.density_from_betas(aic_best_betas)
            aicc_density = self.density_from_betas(aicc_best_betas)
            self.map_density = bic_density
        else:
            aic_density, aicc_density, bic_density = None, None, None
        
        self.map_betas = bic_best_betas

        return {'aic': aic_trace,
                'aicc': aicc_trace,
                'bic': bic_trace,
                'dof': dof_trace,
                'loglikelihood': log_likelihood_trace,
                'lambdas': lambda_grid,
                'aic_betas': aic_best_betas,
                'aicc_betas': aicc_best_betas,
                'bic_betas': bic_best_betas,
                'aic_best_idx': aic_best_idx,
                'aicc_best_idx': aicc_best_idx,
                'bic_best_idx': bic_best_idx,
                'aic_densities': aic_density.reshape(self.data_shape),
                'aicc_densities': aicc_density.reshape(self.data_shape),
                'bic_densities': bic_density.reshape(self.data_shape)}