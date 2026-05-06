def solution_path(self, min_lambda, max_lambda, lambda_bins, verbose=0):
        '''Follows the solution path to find the best lambda value.'''
        self.u = np.zeros(self.Dk.shape[0], dtype='double')
        lambda_grid = np.exp(np.linspace(np.log(max_lambda), np.log(min_lambda), lambda_bins))
        aic_trace = np.zeros(lambda_grid.shape) # The AIC score for each lambda value
        aicc_trace = np.zeros(lambda_grid.shape) # The AICc score for each lambda value (correcting for finite sample size)
        bic_trace = np.zeros(lambda_grid.shape) # The BIC score for each lambda value
        dof_trace = np.zeros(lambda_grid.shape) # The degrees of freedom of each final solution
        log_likelihood_trace = np.zeros(lambda_grid.shape)
        beta_trace = []
        best_idx = None
        best_plateaus = None

        if self.edges is None:
            self.edges = defaultdict(list)
            elist = csr_matrix(self.D).indices.reshape((self.D.shape[0], 2))
            for n1, n2 in elist:
                self.edges[n1].append(n2)
                self.edges[n2].append(n1)

        # Solve the series of lambda values with warm starts at each point
        for i, lam in enumerate(lambda_grid):
            if verbose:
                print('#{0} Lambda = {1}'.format(i, lam))

            # Fit to the final values
            beta = self.solve(lam)

            if verbose:
                print('Calculating degrees of freedom')

            # Count the number of free parameters in the grid (dof) -- TODO: the graph trend filtering paper seems to imply we shouldn't multiply by (k+1)?
            dof_vals = self.Dk_minus_one.dot(beta) if self.k > 0 else beta
            plateaus = calc_plateaus(dof_vals, self.edges, rel_tol=0.01) if (self.k % 2) == 0 else nearly_unique(dof_vals, rel_tol=0.03)
            dof_trace[i] = max(1,len(plateaus)) #* (k+1)

            if verbose:
                print('Calculating Information Criteria')

            # Get the negative log-likelihood
            log_likelihood_trace[i] = -0.5 * ((self.y - beta)**2).sum()

            # Calculate AIC = 2k - 2ln(L)
            aic_trace[i] = 2. * dof_trace[i] - 2. * log_likelihood_trace[i]
            
            # Calculate AICc = AIC + 2k * (k+1) / (n - k - 1)
            aicc_trace[i] = aic_trace[i] + 2 * dof_trace[i] * (dof_trace[i]+1) / (len(beta) - dof_trace[i] - 1.)

            # Calculate BIC = -2ln(L) + k * (ln(n) - ln(2pi))
            bic_trace[i] = -2 * log_likelihood_trace[i] + dof_trace[i] * (np.log(len(beta)) - np.log(2 * np.pi))

            # Track the best model thus far
            if best_idx is None or bic_trace[i] < bic_trace[best_idx]:
                best_idx = i
                best_plateaus = plateaus

            # Save the trace of all the resulting parameters
            beta_trace.append(np.array(beta))
            
            if verbose:
                print('DoF: {0} AIC: {1} AICc: {2} BIC: {3}\n'.format(dof_trace[i], aic_trace[i], aicc_trace[i], bic_trace[i]))

        if verbose:
            print('Best setting (by BIC): lambda={0} [DoF: {1}, AIC: {2}, AICc: {3} BIC: {4}]'.format(lambda_grid[best_idx], dof_trace[best_idx], aic_trace[best_idx], aicc_trace[best_idx], bic_trace[best_idx]))

        return {'aic': aic_trace,
                'aicc': aicc_trace,
                'bic': bic_trace,
                'dof': dof_trace,
                'loglikelihood': log_likelihood_trace,
                'beta': np.array(beta_trace),
                'lambda': lambda_grid,
                'best_idx': best_idx,
                'best': beta_trace[best_idx],
                'plateaus': best_plateaus}