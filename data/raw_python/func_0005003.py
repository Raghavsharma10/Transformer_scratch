def add(self, operator):
        '''Add an operator to the Slicer

        Parameters
        ----------
        operator : Scope (TaskTransformer or FeatureExtractor)
            The new operator to add
        '''
        if not isinstance(operator, Scope):
            raise ParameterError('Operator {} must be a TaskTransformer '
                                 'or FeatureExtractor'.format(operator))
        for key in operator.fields:
            self._time[key] = []
            # We add 1 to the dimension here to account for batching
            for tdim, idx in enumerate(operator.fields[key].shape, 1):
                if idx is None:
                    self._time[key].append(tdim)