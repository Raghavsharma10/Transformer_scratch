def transform_audio(self, y):
        '''Compute the CQT

        Parameters
        ----------
        y : np.ndarray
            The audio buffer

        Returns
        -------
        data : dict
            data['mag'] : np.ndarray, shape = (n_frames, n_bins)
                The CQT magnitude

            data['phase']: np.ndarray, shape = mag.shape
                The CQT phase
        '''
        n_frames = self.n_frames(get_duration(y=y, sr=self.sr))

        C = cqt(y=y, sr=self.sr, hop_length=self.hop_length,
                fmin=self.fmin,
                n_bins=(self.n_octaves * self.over_sample * 12),
                bins_per_octave=(self.over_sample * 12))

        C = fix_length(C, n_frames)

        cqtm, phase = magphase(C)
        if self.log:
            cqtm = amplitude_to_db(cqtm, ref=np.max)

        return {'mag': cqtm.T.astype(np.float32)[self.idx],
                'phase': np.angle(phase).T.astype(np.float32)[self.idx]}