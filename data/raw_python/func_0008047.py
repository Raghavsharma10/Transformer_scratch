def fit(self):
        """
        Maximum-likelihood estimation of the HMM using the Baum-Welch algorithm

        Returns
        -------
        model : HMM
            The maximum likelihood HMM model.

        """
        logger().info("=================================================================")
        logger().info("Running Baum-Welch:")
        logger().info("  input observations: "+str(self.nobservations)+" of lengths "+str(self.observation_lengths))
        logger().info("  initial HMM guess:"+str(self._hmm))

        initial_time = time.time()

        it = 0
        self._likelihoods = np.zeros(self.maxit)
        loglik = 0.0
        # flag if connectivity has changed (e.g. state lost) - in that case the likelihood
        # is discontinuous and can't be used as a convergence criterion in that iteration.
        tmatrix_nonzeros = self.hmm.transition_matrix.nonzero()
        converged = False

        while not converged and it < self.maxit:
            # self._fbtimings = np.zeros(5)
            t1 = time.time()
            loglik = 0.0
            for k in range(self._nobs):
                loglik += self._forward_backward(k)
                assert np.isfinite(loglik), it
            t2 = time.time()

            # convergence check
            if it > 0:
                dL = loglik - self._likelihoods[it-1]
                # print 'dL ', dL, 'iter_P ', maxiter_P
                if dL < self._accuracy:
                    # print "CONVERGED! Likelihood change = ",(loglik - self.likelihoods[it-1])
                    converged = True

            # update model
            self._update_model(self._gammas, self._Cs, maxiter=self._maxit_P)
            t3 = time.time()

            # connectivity change check
            tmatrix_nonzeros_new = self.hmm.transition_matrix.nonzero()
            if not np.array_equal(tmatrix_nonzeros, tmatrix_nonzeros_new):
                converged = False  # unset converged
                tmatrix_nonzeros = tmatrix_nonzeros_new

            # print 't_fb: ', str(1000.0*(t2-t1)), 't_up: ', str(1000.0*(t3-t2)), 'L = ', loglik, 'dL = ', (loglik - self._likelihoods[it-1])
            # print '  fb timings (ms): pobs', (1000.0*self._fbtimings).astype(int)

            logger().info(str(it) + " ll = " + str(loglik))
            # print self.model.output_model
            # print "---------------------"

            # end of iteration
            self._likelihoods[it] = loglik
            it += 1

        # final update with high precision
        # self._update_model(self._gammas, self._Cs, maxiter=10000000)

        # truncate likelihood history
        self._likelihoods = self._likelihoods[:it]
        # set final likelihood
        self._hmm.likelihood = loglik
        # set final count matrix
        self.count_matrix = self._transition_counts(self._Cs)
        self.initial_count = self._init_counts(self._gammas)

        final_time = time.time()
        elapsed_time = final_time - initial_time

        logger().info("maximum likelihood HMM:"+str(self._hmm))
        logger().info("Elapsed time for Baum-Welch solution: %.3f s" % elapsed_time)
        logger().info("Computing Viterbi path:")

        initial_time = time.time()

        # Compute hidden state trajectories using the Viterbi algorithm.
        self._hmm.hidden_state_trajectories = self.compute_viterbi_paths()

        final_time = time.time()
        elapsed_time = final_time - initial_time

        logger().info("Elapsed time for Viterbi path computation: %.3f s" % elapsed_time)
        logger().info("=================================================================")

        return self._hmm