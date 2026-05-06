def plot_series_residuals_internal(self, varied_data, varied_idx, **kwargs):
        """ Analogous to :meth:`plot_series` but for internal residuals from last run. """
        nf = len(self.f_cb(*self.pre_process(
            self.internal_xout[0], self.internal_params_out[0])))
        xerr = np.empty((self.internal_xout.shape[0], nf))
        for idx, (res, params) in enumerate(zip(self.internal_xout, self.internal_params_out)):
            xerr[idx, :] = self.f_cb(res, params)
        return self.plot_series(xerr, varied_data, varied_idx, **kwargs)