def get_eta(self, mag, std):
        """
        Return Eta feature.

        Parameters
        ----------
        mag : array_like
            An array of magnitudes.
        std : array_like
            A standard deviation of magnitudes.

        Returns
        -------
        eta : float
            The value of Eta index.
        """

        diff = mag[1:] - mag[:len(mag) - 1]
        eta = np.sum(diff * diff) / (len(mag) - 1.) / std / std

        return eta