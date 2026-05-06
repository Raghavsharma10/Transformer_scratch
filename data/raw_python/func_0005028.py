def transform_audio(self, y):
        '''Apply the scale transform to the tempogram

        Parameters
        ----------
        y : np.ndarray
            The audio buffer

        Returns
        -------
        data : dict
            data['temposcale'] : np.ndarray, shape=(n_frames, n_fmt)
                The scale transform magnitude coefficients
        '''
        data = super(TempoScale, self).transform_audio(y)
        data['temposcale'] = np.abs(fmt(data.pop('tempogram'),
                                        axis=1,
                                        n_fmt=self.n_fmt)).astype(np.float32)[self.idx]
        return data