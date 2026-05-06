def create_design_matrix(self, demean = False, intercept = True):
        """create_design_matrix calls create_event_regressors for each of the covariates in the self.covariates dict. self.designmatrix is created and is shaped (nr_regressors, self.resampled_signal.shape[-1])
        """
        self.design_matrix = np.zeros((int(self.number_of_event_types*self.deconvolution_interval_size), self.resampled_signal_size))

        for i, covariate in enumerate(self.covariates.keys()):
            # document the creation of the designmatrix step by step
            self.logger.debug('creating regressor for ' + covariate)
            indices = np.arange(i*self.deconvolution_interval_size,(i+1)*self.deconvolution_interval_size, dtype = int)
            # here, we implement the dot-separated encoding of events and covariates
            if len(covariate.split('.')) > 0:
                which_event_time_indices = covariate.split('.')[0]
            else:
                which_event_time_indices = covariate
            self.design_matrix[indices] = self.create_event_regressors( self.event_times_indices[which_event_time_indices], 
                                                                        self.covariates[covariate], 
                                                                        self.durations[which_event_time_indices])

        if demean:
            # we expect the data to be demeaned. 
            # it's an option whether the regressors should be, too
            self.design_matrix = (self.design_matrix.T - self.design_matrix.mean(axis = -1)).T
        if intercept:
            # similarly, intercept is a choice.
            self.design_matrix = np.vstack((self.design_matrix, np.ones((1,self.design_matrix.shape[-1]))))
        
        self.logger.debug('created %s design_matrix' % (str(self.design_matrix.shape)))