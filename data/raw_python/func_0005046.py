def transform_audio(self, y):
        '''Compute the HCQT with unwrapped phase

        Parameters
        ----------
        y : np.ndarray
            The audio buffer

        Returns
        -------
        data : dict
            data['mag'] : np.ndarray, shape=(n_frames, n_bins)
                CQT magnitude

            data['dphase'] : np.ndarray, shape=(n_frames, n_bins)
                Unwrapped phase differential
        '''
        data = super(HCQTPhaseDiff, self).transform_audio(y)
        data['dphase'] = self.phase_diff(data.pop('phase'))
        return data