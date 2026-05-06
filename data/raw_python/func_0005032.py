def transform_audio(self, y):
        '''Compute the STFT

        Parameters
        ----------
        y : np.ndarray
            The audio buffer

        Returns
        -------
        data : dict
            data['mag'] : np.ndarray, shape=(n_frames, 1 + n_fft//2)
                The STFT magnitude
        '''
        data = super(STFTMag, self).transform_audio(y)
        data.pop('phase')

        return data