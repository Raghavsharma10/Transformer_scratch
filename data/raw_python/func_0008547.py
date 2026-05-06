def plot_series_residuals(self, xres, varied_data, varied_idx, params, **kwargs):
        """ Analogous to :meth:`plot_series` but will plot residuals. """
        nf = len(self.f_cb(*self.pre_process(xres[0], params)))
        xerr = np.empty((xres.shape[0], nf))
        new_params = np.array(params)

        for idx, row in enumerate(xres):
            new_params[varied_idx] = varied_data[idx]
            xerr[idx, :] = self.f_cb(*self.pre_process(row, params))
        return self.plot_series(xerr, varied_data, varied_idx, **kwargs)