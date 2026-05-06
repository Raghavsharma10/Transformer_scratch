def encode_intervals(self, duration, intervals, values, dtype=np.bool,
                         multi=True, fill=None):
        '''Encode labeled intervals as a time-series matrix.

        Parameters
        ----------
        duration : number
            The duration (in frames) of the track

        intervals : np.ndarray, shape=(n, 2)
            The list of intervals

        values : np.ndarray, shape=(n, m)
            The (encoded) values corresponding to each interval

        dtype : np.dtype
            The desired output type

        multi : bool
            If `True`, allow multiple labels per interval.

        fill : dtype (optional)
            Optional default fill value for missing data.

            If not provided, the default is inferred from `dtype`.

        Returns
        -------
        target : np.ndarray, shape=(duration * sr / hop_length, m)
            The labeled interval encoding, sampled at the desired frame rate
        '''
        if fill is None:
            fill = fill_value(dtype)

        frames = time_to_frames(intervals, sr=self.sr,
                                hop_length=self.hop_length)

        n_total = int(time_to_frames(duration, sr=self.sr,
                                     hop_length=self.hop_length))

        values = values.astype(dtype)

        n_alloc = n_total
        if np.any(frames):
            n_alloc = max(n_total, 1 + int(frames.max()))

        target = np.empty((n_alloc, values.shape[1]),

                          dtype=dtype)

        target.fill(fill)

        for column, interval in zip(values, frames):
            if multi:
                target[interval[0]:interval[1]] += column
            else:
                target[interval[0]:interval[1]] = column

        return target[:n_total]