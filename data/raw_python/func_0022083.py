def betas_for_cov(self, covariate = '0'):
        """betas_for_cov returns the beta values (i.e. IRF) associated with a specific covariate.

            :param covariate: name of covariate.
            :type covariate: string
        """
        # find the index in the designmatrix of the current covariate
        this_covariate_index = list(self.covariates.keys()).index(covariate)
        return self.betas[int(this_covariate_index*self.deconvolution_interval_size):int((this_covariate_index+1)*self.deconvolution_interval_size)]