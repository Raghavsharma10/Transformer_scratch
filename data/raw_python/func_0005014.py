def transform_annotation(self, ann, duration):
        '''Transform an annotation to static label encoding.

        Parameters
        ----------
        ann : jams.Annotation
            The annotation to convert

        duration : number > 0
            The duration of the track

        Returns
        -------
        data : dict
            data['tags'] : np.ndarray, shape=(n_labels,)
                A static binary encoding of the labels
        '''
        intervals = np.asarray([[0, 1]])
        values = list([obs.value for obs in ann])
        intervals = np.tile(intervals, [len(values), 1])

        # Suppress all intervals not in the encoder
        tags = [v for v in values if v in self._classes]
        if len(tags):
            target = self.encoder.transform([tags]).astype(np.bool).max(axis=0)
        else:
            target = np.zeros(len(self._classes), dtype=np.bool)

        return {'tags': target}