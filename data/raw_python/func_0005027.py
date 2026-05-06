def transform_audio(self, y):
        '''Compute the tempogram

        Parameters
        ----------
        y : np.ndarray
            Audio buffer

        Returns
        -------
        data : dict
            data['tempogram'] : np.ndarray, shape=(n_frames, win_length)
                The tempogram
        '''
        n_frames = self.n_frames(get_duration(y=y, sr=self.sr))

        tgram = tempogram(y=y, sr=self.sr,
                          hop_length=self.hop_length,
                          win_length=self.win_length).astype(np.float32)

        tgram = fix_length(tgram, n_frames)
        return {'tempogram': tgram.T[self.idx]}