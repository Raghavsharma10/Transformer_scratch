def add_continuous_regressors_to_design_matrix(self, regressors):
        """add_continuous_regressors_to_design_matrix appends continuously sampled regressors to the existing design matrix. One uses this addition to the design matrix when one expects the data to contain nuisance factors that aren't tied to the moments of specific events. For instance, in fMRI analysis this allows us to add cardiac / respiratory regressors, as well as tissue and head motion timecourses to the designmatrix.
        
            :param regressors: the signal to be appended to the design matrix.
            :type regressors: numpy array, with shape equal to (nr_regressors, self.resampled_signal.shape[-1])
        """
        previous_design_matrix_shape = self.design_matrix.shape
        if len(regressors.shape) == 1:
            regressors = regressors[np.newaxis, :]
        if regressors.shape[1] != self.resampled_signal.shape[1]:
            self.logger.warning('additional regressor shape %s does not conform to designmatrix shape %s' % (regressors.shape, self.resampled_signal.shape))
        # and, an vstack append
        self.design_matrix = np.vstack((self.design_matrix, regressors))
        self.logger.debug('added %s continuous regressors to %s design_matrix, shape now %s' % (str(regressors.shape), str(previous_design_matrix_shape), str(self.design_matrix.shape)))