def _add_parameter(self, parameter):
        '''
        Force adds a `Parameter` object to the instance.
        '''
        if isinstance(parameter, MethodParameter):
            # create a bound instance of the MethodParameter
            parameter = parameter.bind(alloy=self)
        self._parameters[parameter.name] = parameter
        for alias in parameter.aliases:
            self._aliases[alias] = parameter