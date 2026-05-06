def spectrum(self, ref=None, segmentLengthMultiplier=1, mode=None, **kwargs):
        """
        analyses the source to generate the linear spectrum.
        :param ref: the reference value for dB purposes.
        :param segmentLengthMultiplier: allow for increased resolution.
        :param mode: cq or none.
        :return:
            f : ndarray
            Array of sample frequencies.
            Pxx : ndarray
            linear spectrum.

        """

        def analysisFunc(x, nperseg, **kwargs):
            f, Pxx_spec = signal.welch(self.samples, self.fs, nperseg=nperseg, scaling='spectrum', detrend=False,
                                       **kwargs)
            Pxx_spec = np.sqrt(Pxx_spec)
            # it seems a 3dB adjustment is required to account for the change in nperseg
            if x > 0:
                Pxx_spec = Pxx_spec / (10 ** ((3 * x) / 20))
            if ref is not None:
                Pxx_spec = librosa.amplitude_to_db(Pxx_spec, ref)
            return f, Pxx_spec

        if mode == 'cq':
            return self._cq(analysisFunc, segmentLengthMultiplier)
        else:
            return analysisFunc(0, self.getSegmentLength() * segmentLengthMultiplier, **kwargs)