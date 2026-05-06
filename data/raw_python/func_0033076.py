def _get_z(self, kwargs):
        '''
        Returns z if type is 1 or 2 and z is explicitly defined in kwargs.
        Otherwise, raises TypeError.
        '''
        if self._type == 1 or self._type == 2:
            if 'z' in kwargs:
                return round(float(kwargs['z']), 6)
            elif self._element_z in kwargs:
                return round(float(kwargs[self._element_z]), 6)
        raise TypeError()