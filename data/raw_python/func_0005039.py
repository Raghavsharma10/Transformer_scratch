def transform_annotation(self, ann, duration):
        '''Transform an annotation to chord-tag encoding

        Parameters
        ----------
        ann : jams.Annotation
            The annotation to convert

        duration : number > 0
            The duration of the track

        Returns
        -------
        data : dict
            data['chord'] : np.ndarray, shape=(n, n_labels)
                A time-varying binary encoding of the chords
        '''

        intervals, values = ann.to_interval_values()

        chords = []
        for v in values:
            chords.extend(self.encoder.transform([self.simplify(v)]))

        dtype = self.fields[self.scope('chord')].dtype

        chords = np.asarray(chords)

        if self.sparse:
            chords = chords[:, np.newaxis]

        target = self.encode_intervals(duration, intervals, chords,
                                       multi=False, dtype=dtype)

        return {'chord': target}