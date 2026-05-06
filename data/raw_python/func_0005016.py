def transform_audio(self, y):
        '''Compute the time position encoding

        Parameters
        ----------
        y : np.ndarray
            Audio buffer

        Returns
        -------
        data : dict
            data['relative'] = np.ndarray, shape=(n_frames, 2)
            data['absolute'] = np.ndarray, shape=(n_frames, 2)

                Relative and absolute time positional encodings.
        '''

        duration = get_duration(y=y, sr=self.sr)
        n_frames = self.n_frames(duration)

        relative = np.zeros((n_frames, 2), dtype=np.float32)
        relative[:, 0] = np.cos(np.pi * np.linspace(0, 1, num=n_frames))
        relative[:, 1] = np.sin(np.pi * np.linspace(0, 1, num=n_frames))

        absolute = relative * np.sqrt(duration)

        return {'relative': relative[self.idx],
                'absolute': absolute[self.idx]}