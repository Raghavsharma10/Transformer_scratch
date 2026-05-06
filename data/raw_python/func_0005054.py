def transform(self, y, sr):
        '''Transform an audio signal

        Parameters
        ----------
        y : np.ndarray
            The audio signal

        sr : number > 0
            The native sampling rate of y

        Returns
        -------
        dict
            Data dictionary containing features extracted from y

        See Also
        --------
        transform_audio
        '''
        if sr != self.sr:
            y = resample(y, sr, self.sr)

        return self.merge([self.transform_audio(y)])