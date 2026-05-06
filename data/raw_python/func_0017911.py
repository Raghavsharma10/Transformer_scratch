def get_stetson_k(self, mag, avg, err):
        """
        Return Stetson K feature.

        Parameters
        ----------
        mag : array_like
            An array of magnitude.
        avg : float
            An average value of magnitudes.
        err : array_like
            An array of magnitude errors.

        Returns
        -------
        stetson_k : float
            Stetson K value.
        """

        residual = (mag - avg) / err
        stetson_k = np.sum(np.fabs(residual)) \
                    / np.sqrt(np.sum(residual * residual)) / np.sqrt(len(mag))

        return stetson_k