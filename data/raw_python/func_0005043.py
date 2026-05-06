def transform_audio(self, y):
        '''Compute the HCQT

        Parameters
        ----------
        y : np.ndarray
            The audio buffer

        Returns
        -------
        data : dict
            data['mag'] : np.ndarray, shape = (n_frames, n_bins, n_harmonics)
                The CQT magnitude

            data['phase']: np.ndarray, shape = mag.shape
                The CQT phase
        '''
        cqtm, phase = [], []

        n_frames = self.n_frames(get_duration(y=y, sr=self.sr))

        for h in self.harmonics:
            C = cqt(y=y, sr=self.sr, hop_length=self.hop_length,
                    fmin=self.fmin * h,
                    n_bins=(self.n_octaves * self.over_sample * 12),
                    bins_per_octave=(self.over_sample * 12))

            C = fix_length(C, n_frames)

            C, P = magphase(C)
            if self.log:
                C = amplitude_to_db(C, ref=np.max)
            cqtm.append(C)
            phase.append(P)

        cqtm = np.asarray(cqtm).astype(np.float32)
        phase = np.angle(np.asarray(phase)).astype(np.float32)

        return {'mag': self._index(cqtm),
                'phase': self._index(phase)}