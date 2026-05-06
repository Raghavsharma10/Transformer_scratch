def transform_annotation(self, ann, duration):
        '''Transform an annotation to dynamic label encoding.

        Parameters
        ----------
        ann : jams.Annotation
            The annotation to convert

        duration : number > 0
            The duration of the track

        Returns
        -------
        data : dict
            data['tags'] : np.ndarray, shape=(n, n_labels)
                A time-varying binary encoding of the labels
        '''
        intervals, values = ann.to_interval_values()

        # Suppress all intervals not in the encoder
        tags = []
        for v in values:
            if v in self._classes:
                tags.extend(self.encoder.transform([[v]]))
            else:
                tags.extend(self.encoder.transform([[]]))

        tags = np.asarray(tags)
        target = self.encode_intervals(duration, intervals, tags)

        return {'tags': target}