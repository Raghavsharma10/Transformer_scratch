def _get_x(self, kwargs):
        '''
        Returns x if it is explicitly defined in kwargs.
        Otherwise, raises TypeError.
        '''
        if 'x' in kwargs:
            return round(float(kwargs['x']), 6)
        elif self._element_x in kwargs:
            return round(float(kwargs[self._element_x]), 6)
        elif self._type == 3 and self._element_1mx in kwargs:
            return round(1. - float(kwargs[self._element_1mx]), 6)
        else:
            raise TypeError()