def add(self, operator):
        '''Add an operation to this pump.

        Parameters
        ----------
        operator : BaseTaskTransformer, FeatureExtractor
            The operation to add

        Raises
        ------
        ParameterError
            if `op` is not of a correct type
        '''
        if not isinstance(operator, (BaseTaskTransformer, FeatureExtractor)):
            raise ParameterError('operator={} must be one of '
                                 '(BaseTaskTransformer, FeatureExtractor)'
                                 .format(operator))

        if operator.name in self.opmap:
            raise ParameterError('Duplicate operator name detected: '
                                 '{}'.format(operator))

        super(Pump, self).add(operator)
        self.opmap[operator.name] = operator
        self.ops.append(operator)