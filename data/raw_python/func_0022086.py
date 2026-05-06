def calculate_rsq(self):
        """calculate_rsq calculates coefficient of determination, or r-squared, defined here as 1.0 - SS_res / SS_tot. rsq is only calculated for those timepoints in the data for which the design matrix is non-zero.
        """
        assert hasattr(self, 'betas'), 'no betas found, please run regression before rsq'

        explained_times = self.design_matrix.sum(axis = 0) != 0

        explained_signal = self.predict_from_design_matrix(self.design_matrix)
        self.rsq = 1.0 - np.sum((explained_signal[:,explained_times] - self.resampled_signal[:,explained_times])**2, axis = -1) / np.sum(self.resampled_signal[:,explained_times].squeeze()**2, axis = -1)
        self.ssr = np.sum((explained_signal[:,explained_times] - self.resampled_signal[:,explained_times])**2, axis = -1)
        return np.squeeze(self.rsq)