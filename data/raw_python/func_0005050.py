def encode_events(self, duration, events, values, dtype=np.bool):
        '''Encode labeled events as a time-series matrix.

        Parameters
        ----------
        duration : number
            The duration of the track

        events : ndarray, shape=(n,)
            Time index of the events

        values : ndarray, shape=(n, m)
            Values array.  Must have the same first index as `events`.

        dtype : numpy data type

        Returns
        -------
        target : ndarray, shape=(n_frames, n_values)
        '''

        frames = time_to_frames(events, sr=self.sr,
                                hop_length=self.hop_length)

        n_total = int(time_to_frames(duration, sr=self.sr,
                                     hop_length=self.hop_length))

        n_alloc = n_total
        if np.any(frames):
            n_alloc = max(n_total, 1 + int(frames.max()))

        target = np.empty((n_alloc, values.shape[1]),
                          dtype=dtype)

        target.fill(fill_value(dtype))
        values = values.astype(dtype)
        for column, event in zip(values, frames):
            target[event] += column

        return target[:n_total]