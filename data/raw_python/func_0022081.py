def regress(self, method = 'lstsq'):
        """regress performs linear least squares regression of the designmatrix on the data. 

            :param method: method, or backend to be used for the regression analysis.
            :type method: string, one of ['lstsq', 'sm_ols']
            :returns: instance variables 'betas' (nr_betas x nr_signals) and 'residuals' (nr_signals x nr_samples) are created.
        """

        if method is 'lstsq':
            self.betas, residuals_sum, rank, s = LA.lstsq(self.design_matrix.T, self.resampled_signal.T)
            self.residuals = self.resampled_signal - self.predict_from_design_matrix(self.design_matrix)
        elif method is 'sm_ols':
            import statsmodels.api as sm

            assert self.resampled_signal.shape[0] == 1, \
                    'signal input into statsmodels OLS cannot contain multiple signals at once, present shape %s' % str(self.resampled_signal.shape)
            model = sm.OLS(np.squeeze(self.resampled_signal),self.design_matrix.T)
            results = model.fit()
            # make betas and residuals that are compatible with the LA.lstsq type.
            self.betas = np.array(results.params).reshape((self.design_matrix.shape[0], self.resampled_signal.shape[0]))
            self.residuals = np.array(results.resid).reshape(self.resampled_signal.shape)

        self.logger.debug('performed %s regression on %s design_matrix and %s signal' % (method, str(self.design_matrix.shape), str(self.resampled_signal.shape)))