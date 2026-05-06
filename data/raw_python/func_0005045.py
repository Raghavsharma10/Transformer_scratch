def transform_audio(self, y):
        '''Compute HCQT magnitude.

        Parameters
        ----------
        y : np.ndarray
            the audio buffer

        Returns
        -------
        data : dict
            data['mag'] : np.ndarray, shape=(n_frames, n_bins)
                The CQT magnitude
        '''
        data = super(HCQTMag, self).transform_audio(y)
        data.pop('phase')
        return data