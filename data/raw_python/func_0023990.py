def compute_K_L_alpha_ll(self):
        r"""Compute `K`, `L`, `alpha` and log-likelihood according to the first part of Algorithm 2.1 in R&W.
        
        Computes `K` and the noise portion of `K` using :py:meth:`compute_Kij`,
        computes `L` using :py:func:`scipy.linalg.cholesky`, then computes
        `alpha` as `L.T\\(L\\y)`.
        
        Only does the computation if :py:attr:`K_up_to_date` is False --
        otherwise leaves the existing values.
        """
        if not self.K_up_to_date:
            y = self.y
            err_y = self.err_y
            self.K = self.compute_Kij(self.X, None, self.n, None, noise=False)
            # If the noise kernel is meant to be strictly diagonal, it should
            # yield a diagonal noise_K:
            if isinstance(self.noise_k, ZeroKernel):
                self.noise_K = scipy.zeros((self.X.shape[0], self.X.shape[0]))
            elif isinstance(self.noise_k, DiagonalNoiseKernel):
                self.noise_K = self.noise_k.params[0]**2.0 * scipy.eye(self.X.shape[0])
            else:
                self.noise_K = self.compute_Kij(self.X, None, self.n, None, noise=True)
            
            K = self.K
            noise_K = self.noise_K
            if self.T is not None:
                KnK = self.T.dot(K + noise_K).dot(self.T.T)
            else:
                KnK = K + noise_K
            K_tot = (
                KnK +
                scipy.diag(err_y**2.0) +
                self.diag_factor * sys.float_info.epsilon * scipy.eye(len(y))
            )
            self.L = scipy.linalg.cholesky(K_tot, lower=True)
            # Need to make the mean-subtracted y that appears in the expression
            # for alpha:
            if self.mu is not None:
                mu_alph = self.mu(self.X, self.n)
                if self.T is not None:
                    mu_alph = self.T.dot(mu_alph)
                y_alph = self.y - mu_alph
            else:
                y_alph = self.y
            self.alpha = scipy.linalg.cho_solve((self.L, True), scipy.atleast_2d(y_alph).T)
            self.ll = (
                -0.5 * scipy.atleast_2d(y_alph).dot(self.alpha) -
                scipy.log(scipy.diag(self.L)).sum() - 
                0.5 * len(y) * scipy.log(2.0 * scipy.pi)
            )[0, 0]
            # Apply hyperpriors:
            self.ll += self.hyperprior(self.params)
            
            if self.use_hyper_deriv:
                warnings.warn("Use of hyperparameter derivatives is experimental!")
                
                # Only compute for the free parameters, since that is what we
                # want to optimize:
                self.ll_deriv = scipy.zeros(len(self.free_params))
                # Combine the kernel and noise kernel so we only need one loop:
                if isinstance(self.noise_k, ZeroKernel):
                    knk = self.k
                elif isinstance(self.noise_k, DiagonalNoiseKernel):
                    knk = self.k
                    # Handle DiagonalNoiseKernel specially:
                    if not self.noise_k.fixed_params[0]:
                        dK_dtheta_i = 2.0 * self.noise_k.params[0] * scipy.eye(len(y))
                        self.ll_deriv[len(self.k.free_params)] = 0.5 * (
                            self.alpha.T.dot(dK_dtheta_i.dot(self.alpha)) -
                            scipy.trace(scipy.linalg.cho_solve((self.L, True), dK_dtheta_i))
                        )
                else:
                    knk = self.k + self.noise_k
                
                # Get the indices of the free params in knk.params:
                free_param_idxs = scipy.arange(0, len(knk.params), dtype=int)[~knk.fixed_params]
                # Handle the kernel and noise kernel:
                for i, pi in enumerate(free_param_idxs):
                    dK_dtheta_i = self.compute_Kij(
                        self.X, None, self.n, None, k=knk, hyper_deriv=pi
                    )
                    if self.T is not None:
                        dK_dtheta_i = self.T.dot(dK_dtheta_i).dot(self.T.T)
                    self.ll_deriv[i] = 0.5 * (
                        self.alpha.T.dot(dK_dtheta_i.dot(self.alpha)) -
                        scipy.trace(scipy.linalg.cho_solve((self.L, True), dK_dtheta_i))
                    )
                
                # Handle the mean function:
                if self.mu is not None:
                    # Get the indices of the free params in self.mu.params:
                    free_param_idxs = scipy.arange(0, len(self.mu.params), dtype=int)[~self.mu.fixed_params]
                    for i, pi in enumerate(free_param_idxs):
                        dmu_dtheta_i = scipy.atleast_2d(self.mu(self.X, self.n, hyper_deriv=pi)).T
                        if self.T is not None:
                            dmu_dtheta_i = self.T.dot(dmu_dtheta_i)
                        self.ll_deriv[i + len(knk.free_params)] = dmu_dtheta_i.T.dot(self.alpha)
                
                # Handle the hyperprior:
                # Get the indices of the free params in self.params:
                free_param_idxs = scipy.arange(0, len(self.params), dtype=int)[~self.fixed_params]
                for i, pi in enumerate(free_param_idxs):
                    self.ll_deriv[i] += self.hyperprior(self.params, hyper_deriv=pi)
            
            self.K_up_to_date = True