def predict_from_design_matrix(self, design_matrix):
        """predict_from_design_matrix predicts signals given a design matrix.

            :param design_matrix: design matrix from which to predict a signal.
            :type design_matrix: numpy array, (nr_samples x betas.shape)
            :returns: predicted signal(s) 
            :rtype: numpy array (nr_signals x nr_samples)
        """
        # check if we have already run the regression - which is necessary
        assert hasattr(self, 'betas'), 'no betas found, please run regression before prediction'
        assert design_matrix.shape[0] == self.betas.shape[0], \
                    'designmatrix needs to have the same number of regressors as the betas already calculated'

        # betas = np.copy(self.betas.T, order="F", dtype = np.float32)
        # f_design_matrix = np.copy(design_matrix, order = "F", dtype = np.float32)

        prediction = np.dot(self.betas.astype(np.float32).T, design_matrix.astype(np.float32))

        return prediction