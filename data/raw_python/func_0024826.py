def _process_neg_flux(self, x, y):
        """Remove negative flux."""

        if self._keep_neg:  # Nothing to do
            return y

        old_y = None

        if np.isscalar(y):  # pragma: no cover
            if y < 0:
                n_neg = 1
                old_x = x
                old_y = y
                y = 0
        else:
            x = np.asarray(x)  # In case input is just pure list
            y = np.asarray(y)
            i = np.where(y < 0)
            n_neg = len(i[0])
            if n_neg > 0:
                old_x = x[i]
                old_y = y[i]
                y[i] = 0

        if old_y is not None:
            warn_str = ('{0} bin(s) contained negative flux or throughput'
                        '; it/they will be set to zero.'.format(n_neg))
            warn_str += '\n  points: {0}\n  lookup_table: {1}'.format(
                old_x, old_y)  # Extra info
            self.meta['warnings'].update({'NegativeFlux': warn_str})
            warnings.warn(warn_str, AstropyUserWarning)

        return y