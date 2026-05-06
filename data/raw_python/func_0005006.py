def transform_audio(self, y):
        '''Compute the Mel spectrogram

        Parameters
        ----------
        y : np.ndarray
            The audio buffer

        Returns
        -------
        data : dict
            data['mag'] : np.ndarray, shape=(n_frames, n_mels)
                The Mel spectrogram
        '''
        n_frames = self.n_frames(get_duration(y=y, sr=self.sr))

        mel = np.sqrt(melspectrogram(y=y, sr=self.sr,
                                     n_fft=self.n_fft,
                                     hop_length=self.hop_length,
                                     n_mels=self.n_mels,
                                     fmax=self.fmax)).astype(np.float32)

        mel = fix_length(mel, n_frames)

        if self.log:
            mel = amplitude_to_db(mel, ref=np.max)

        return {'mag': mel.T[self.idx]}