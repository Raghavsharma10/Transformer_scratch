def bootstrap_on_residuals(self, nr_repetitions = 1000):
        """bootstrap_on_residuals bootstraps, by shuffling the residuals. bootstrap_on_residuals should only be used on single-channel data, as otherwise the memory load might increase too much. This uses the lstsq backend regression for a single-pass fit across repetitions. Please note that shuffling the residuals may change the autocorrelation of the bootstrap samples relative to that of the original data and that may reduce its validity. Reference: https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Resampling_residuals

            :param nr_repetitions: number of repetitions for the bootstrap.
            :type nr_repetitions: int

        """
        assert self.resampled_signal.shape[0] == 1, \
                    'signal input into bootstrap_on_residuals cannot contain signals from multiple channels at once, present shape %s' % str(self.resampled_signal.shape)
        assert hasattr(self, 'betas'), 'no betas found, please run regression before bootstrapping'

        # create bootstrap data by taking the residuals
        bootstrap_data = np.zeros((self.resampled_signal_size, nr_repetitions))
        explained_signal = self.predict_from_design_matrix(self.design_matrix).T

        for x in range(bootstrap_data.shape[-1]): # loop over bootstrapsamples
            bootstrap_data[:,x] = (self.residuals.T[np.random.permutation(self.resampled_signal_size)] + explained_signal).squeeze()

        self.bootstrap_betas, bs_residuals, rank, s = LA.lstsq(self.design_matrix.T, bootstrap_data)

        self.bootstrap_betas_per_event_type = np.zeros((len(self.covariates), self.deconvolution_interval_size, nr_repetitions))

        for i, covariate in enumerate(list(self.covariates.keys())):
            # find the index in the designmatrix of the current covariate
            this_covariate_index = list(self.covariates.keys()).index(covariate)
            self.bootstrap_betas_per_event_type[i] = self.bootstrap_betas[this_covariate_index*self.deconvolution_interval_size:(this_covariate_index+1)*self.deconvolution_interval_size]