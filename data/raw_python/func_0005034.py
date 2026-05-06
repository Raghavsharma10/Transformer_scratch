def empty(self, duration):
        '''Empty chord annotations

        Parameters
        ----------
        duration : number
            The length (in seconds) of the empty annotation

        Returns
        -------
        ann : jams.Annotation
            A chord annotation consisting of a single `no-chord` observation.
        '''
        ann = super(ChordTransformer, self).empty(duration)

        ann.append(time=0,
                   duration=duration,
                   value='N', confidence=0)

        return ann