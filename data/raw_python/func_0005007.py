def empty(self, duration):
        '''Empty vector annotations.

        This returns an annotation with a single observation
        vector consisting of all-zeroes.

        Parameters
        ----------
        duration : number >0
            Length of the track

        Returns
        -------
        ann : jams.Annotation
            The empty annotation
        '''
        ann = super(VectorTransformer, self).empty(duration)

        ann.append(time=0, duration=duration, confidence=0,
                   value=np.zeros(self.dimension, dtype=np.float32))
        return ann