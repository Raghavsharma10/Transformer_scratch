def _compute_centers(self, X, sparse, rs):
        """Generate centers, then compute tau, dF and dN vals"""

        super(GRBFRandomLayer, self)._compute_centers(X, sparse, rs)

        centers = self.components_['centers']
        sorted_distances = np.sort(squareform(pdist(centers)))
        self.dF_vals = sorted_distances[:, -1]
        self.dN_vals = sorted_distances[:, 1]/100.0
        #self.dN_vals = 0.0002 * np.ones(self.dF_vals.shape)

        tauNum = np.log(np.log(self.grbf_lambda) /
                        np.log(1.0 - self.grbf_lambda))

        tauDenom = np.log(self.dF_vals/self.dN_vals)

        self.tau_vals = tauNum/tauDenom

        self._extra_args['taus'] = self.tau_vals