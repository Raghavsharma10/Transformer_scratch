def transform_annotation(self, ann, duration):
        '''Transform an annotation to the beat-position encoding

        Parameters
        ----------
        ann : jams.Annotation
            The annotation to convert

        duration : number > 0
            The duration of the track

        Returns
        -------
        data : dict
            data['position'] : np.ndarray, shape=(n, n_labels) or (n, 1)
                A time-varying label encoding of beat position
        '''

        # 1. get all the events
        # 2. find all the downbeats
        # 3. map each downbeat to a subdivision counter
        #       number of beats until the next downbeat
        # 4. pad out events to intervals
        # 5. encode each beat interval to its position

        boundaries, values = ann.to_interval_values()
        # Convert to intervals and span the duration
        # padding at the end of track does not propagate the right label
        # this is an artifact of inferring end-of-track from boundaries though
        boundaries = list(boundaries[:, 0])
        if boundaries and boundaries[-1] < duration:
            boundaries.append(duration)
        intervals = boundaries_to_intervals(boundaries)
        intervals, values = adjust_intervals(intervals, values,
                                             t_min=0,
                                             t_max=duration,
                                             start_label=0,
                                             end_label=0)

        values = np.asarray(values, dtype=int)
        downbeats = np.flatnonzero(values == 1)

        position = []
        for i, v in enumerate(values):
            # If the value is a 0, mark it as X and move on
            if v == 0:
                position.extend(self.encoder.transform(['X']))
                continue

            # Otherwise, let's try to find the surrounding downbeats
            prev_idx = np.searchsorted(downbeats, i, side='right') - 1
            next_idx = 1 + prev_idx

            if prev_idx >= 0 and next_idx < len(downbeats):
                # In this case, the subdivision is well-defined
                subdivision = downbeats[next_idx] - downbeats[prev_idx]
            elif prev_idx < 0 and next_idx < len(downbeats):
                subdivision = np.max(values[:downbeats[0]+1])
            elif next_idx >= len(downbeats):
                subdivision = len(values) - downbeats[prev_idx]

            if subdivision > self.max_divisions or subdivision < 1:
                position.extend(self.encoder.transform(['X']))
            else:
                position.extend(self.encoder.transform(['{:02d}/{:02d}'.format(subdivision, v)]))

        dtype = self.fields[self.scope('position')].dtype

        position = np.asarray(position)
        if self.sparse:
            position = position[:, np.newaxis]

        target = self.encode_intervals(duration, intervals, position,
                                       multi=False, dtype=dtype)
        return {'position': target}