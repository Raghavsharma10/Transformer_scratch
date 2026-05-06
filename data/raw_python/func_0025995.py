def peakSpectrum(self, ref=None, segmentLengthMultiplier=1, mode=None, window='hann'):
        """
        analyses the source to generate the max values per bin per segment
        :param ref: the reference value for dB purposes.
        :param segmentLengthMultiplier: allow for increased resolution.
        :param mode: cq or none.
        :return:
            f : ndarray
            Array of sample frequencies.
            Pxx : ndarray
            linear spectrum max values.
        """

        def analysisFunc(x, nperseg):
            freqs, _, Pxy = signal.spectrogram(self.samples,
                                               self.fs,
                                               window=window,
                                               nperseg=int(nperseg),
                                               noverlap=int(nperseg // 2),
                                               detrend=False,
                                               scaling='spectrum')
            Pxy_max = np.sqrt(Pxy.max(axis=-1).real)
            if x > 0:
                Pxy_max = Pxy_max / (10 ** ((3 * x) / 20))
            if ref is not None:
                Pxy_max = librosa.amplitude_to_db(Pxy_max, ref=ref)
            return freqs, Pxy_max

        if mode == 'cq':
            return self._cq(analysisFunc, segmentLengthMultiplier)
        else:
            return analysisFunc(0, self.getSegmentLength() * segmentLengthMultiplier)