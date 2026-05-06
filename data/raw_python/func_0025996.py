def spectrogram(self, ref=None, segmentLengthMultiplier=1, window='hann'):
        """
        analyses the source to generate a spectrogram
        :param ref: the reference value for dB purposes.
        :param segmentLengthMultiplier: allow for increased resolution.
        :return:
            t : ndarray
            Array of time slices.
            f : ndarray
            Array of sample frequencies.
            Pxx : ndarray
            linear spectrum values.
        """
        t, f, Sxx = signal.spectrogram(self.samples,
                                       self.fs,
                                       window=window,
                                       nperseg=self.getSegmentLength() * segmentLengthMultiplier,
                                       detrend=False,
                                       scaling='spectrum')
        Sxx = np.sqrt(Sxx)
        if ref is not None:
            Sxx = librosa.amplitude_to_db(Sxx, ref)
        return t, f, Sxx