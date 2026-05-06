def transform_annotation(self, ann, duration):
        '''Apply the chord transformation.

        Parameters
        ----------
        ann : jams.Annotation
            The chord annotation

        duration : number > 0
            The target duration

        Returns
        -------
        data : dict
            data['pitch'] : np.ndarray, shape=(n, 12)

            `pitch` is a binary matrix indicating pitch class
            activation at each frame.
        '''
        data = super(SimpleChordTransformer,
                     self).transform_annotation(ann, duration)

        data.pop('root', None)
        data.pop('bass', None)
        return data