def psd(self, ref=None, segmentLengthMultiplier=1, mode=None, **kwargs):
        """
        analyses the source and returns a PSD, segment is set to get ~1Hz frequency resolution
        :param ref: the reference value for dB purposes.
        :param segmentLengthMultiplier: allow for increased resolution.
        :param mode: cq or none.
        :return:
            f : ndarray
            Array of sample frequencies.
            Pxx : ndarray
            Power spectral density.

        """

        def analysisFunc(x, nperseg, **kwargs):
            f, Pxx_den = signal.welch(self.samples, self.fs, nperseg=nperseg, detrend=False, **kwargs)
            if ref is not None:
                Pxx_den = librosa.power_to_db(Pxx_den, ref)
            return f, Pxx_den

        if mode == 'cq':
            return self._cq(analysisFunc, segmentLengthMultiplier)
        else:
            return analysisFunc(0, self.getSegmentLength() * segmentLengthMultiplier, **kwargs)