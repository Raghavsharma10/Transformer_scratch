def ridge_regress(self, cv = 20, alphas = None ):
        """perform k-folds cross-validated ridge regression on the design_matrix. To be used when the design matrix contains very collinear regressors. For cross-validation and ridge fitting, we use sklearn's RidgeCV functionality. Note: intercept is not fit, and data are not prenormalized. 

            :param cv: cross-validated folds, inherits RidgeCV cv argument's functionality.
            :type cv: int, standard = 20
            :param alphas: values of penalization parameter to be traversed by the procedure, inherits RidgeCV cv argument's functionality. Standard value, when parameter is None, is np.logspace(7, 0, 20)
            :type alphas: numpy array, from >0 to 1. 
            :returns: instance variables 'betas' (nr_betas x nr_signals) and 'residuals' (nr_signals x nr_samples) are created.
        """
        if alphas is None:
            alphas = np.logspace(7, 0, 20)
        self.rcv = linear_model.RidgeCV(alphas=alphas, 
                fit_intercept=False, 
                cv=cv) 
        self.rcv.fit(self.design_matrix.T, self.resampled_signal.T)

        self.betas = self.rcv.coef_.T
        self.residuals = self.resampled_signal - self.rcv.predict(self.design_matrix.T)

        self.logger.debug('performed ridge regression on %s design_matrix and %s signal, resulting alpha value is %f' % (str(self.design_matrix.shape), str(self.resampled_signal.shape), self.rcv.alpha_))