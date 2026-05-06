def transform_annotation(self, ann, duration):
        '''Apply the vector transformation.

        Parameters
        ----------
        ann : jams.Annotation
            The input annotation

        duration : number > 0
            The duration of the track

        Returns
        -------
        data : dict
            data['vector'] : np.ndarray, shape=(dimension,)

        Raises
        ------
        DataError
            If the input dimension does not match
        '''
        _, values = ann.to_interval_values()
        vector = np.asarray(values[0], dtype=self.dtype)
        if len(vector) != self.dimension:
            raise DataError('vector dimension({:0}) '
                            '!= self.dimension({:1})'
                            .format(len(vector), self.dimension))

        return {'vector': vector}