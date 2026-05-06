def empty(self, duration):
        '''Empty label annotations.

        Constructs a single observation with an empty value (None).

        Parameters
        ----------
        duration : number > 0
            The duration of the annotation
        '''
        ann = super(DynamicLabelTransformer, self).empty(duration)
        ann.append(time=0, duration=duration, value=None)
        return ann