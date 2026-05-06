def init_kernel(self):
        '''
        Initializes the covariance matrix with a guess at
        the GP kernel parameters.

        '''

        if self.kernel_params is None:
            X = self.apply_mask(self.fpix / self.flux.reshape(-1, 1))
            y = self.apply_mask(self.flux) - np.dot(X, np.linalg.solve(
                np.dot(X.T, X), np.dot(X.T, self.apply_mask(self.flux))))
            white = np.nanmedian([np.nanstd(c) for c in Chunks(y, 13)])
            amp = self.gp_factor * np.nanstd(y)
            tau = 30.0
            if self.kernel == 'Basic':
                self.kernel_params = [white, amp, tau]
            elif self.kernel == 'QuasiPeriodic':
                self.kernel_params = [white, amp, 1., 20.]