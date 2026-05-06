def _estimate_bkg_rms(self, xmin, xmax, ymin, ymax):
        """
        Estimate the background noise mean and RMS.
        The mean is estimated as the median of data.
        The RMS is estimated as the IQR of data / 1.34896.

        Parameters
        ----------
        xmin, xmax, ymin, ymax : int
            The bounding region over which the bkg/rms will be calculated.

        Returns
        -------
        ymin, ymax, xmin, xmax : int
            A copy of the input parameters

        bkg, rms : float
            The calculated background and noise.
        """
        data = self.global_data.data_pix[ymin:ymax, xmin:xmax]
        pixels = np.extract(np.isfinite(data), data).ravel()
        if len(pixels) < 4:
            bkg, rms = np.NaN, np.NaN
        else:
            pixels.sort()
            p25 = pixels[int(pixels.size / 4)]
            p50 = pixels[int(pixels.size / 2)]
            p75 = pixels[int(pixels.size / 4 * 3)]
            iqr = p75 - p25
            bkg, rms = p50, iqr / 1.34896
        # return the input and output data so we know what we are doing
        # when compiling the results of multiple processes
        return ymin, ymax, xmin, xmax, bkg, rms