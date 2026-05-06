def transform_audio(self, y):
        '''Compute the STFT with phase differentials.

        Parameters
        ----------
        y : np.ndarray
            the audio buffer

        Returns
        -------
        data : dict
            data['mag'] : np.ndarray, shape=(n_frames, 1 + n_fft//2)
                The STFT magnitude

            data['dphase'] : np.ndarray, shape=(n_frames, 1 + n_fft//2)
                The unwrapped phase differential
        '''
        data = super(STFTPhaseDiff, self).transform_audio(y)
        data['dphase'] = self.phase_diff(data.pop('phase'))
        return data