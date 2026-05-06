def validation_scatter(self, log_lam, b, masks, pre_v, gp, flux,
                           time, med):
        '''
        Computes the scatter in the validation set.

        '''

        # Update the lambda matrix
        self.lam[b] = 10 ** log_lam

        # Validation set scatter
        scatter = [None for i in range(len(masks))]
        for i in range(len(masks)):
            model = self.cv_compute(b, *pre_v[i])
            try:
                gpm, _ = gp.predict(flux - model - med, time[masks[i]])
            except ValueError:
                # Sometimes the model can have NaNs if
                # `lambda` is a crazy value
                return 1.e30
            fdet = (flux - model)[masks[i]] - gpm
            scatter[i] = 1.e6 * (1.4826 * np.nanmedian(np.abs(fdet / med -
                                 np.nanmedian(fdet / med))) /
                                 np.sqrt(len(masks[i])))

        return np.max(scatter)