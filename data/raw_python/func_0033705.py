def add_parameter(self, parameter, overload=False):
        '''
        Adds a `Parameter` object to the instance.
        
        If a `Parameter` with the same name or alias has already been added
        and `overload` is False (the default), a `ValueError` is thrown.
        
        If a class member or method with the same name or alias is already
        defined, a `ValueError` is thrown, regardless of the value of overload.
        '''
        if not isinstance(parameter, Parameter):
            raise TypeError('`parameter` must be an instance of `Parameter`')

        if hasattr(self, parameter.name):
            item = getattr(self, parameter.name)
            if not isinstance(item, Parameter):
                raise ValueError('"{}" is already a class member or method.'
                                 ''.format(parameter.name))
            elif not overload:
                raise ValueError('Parameter "{}" has already been added'
                                 ' and overload is False.'
                                 ''.format(parameter.name))
        if parameter.name in self._parameters and not overload:
            raise ValueError('Parameter "{}" has already been added'
                             ' and overload is False.'
                             ''.format(parameter.name))
        for alias in parameter.aliases:
            if alias in self._aliases and not overload:
                raise ValueError('Alias "{}" has already been added'
                                 ' and overload is False.'
                                 ''.format(parameter.name))
        self._add_parameter(parameter)