def _get_y(self, kwargs):
        '''
        Returns y if it is explicitly defined in kwargs.
        Otherwise, raises TypeError.
        '''
        if 'y' in kwargs:
            return round(float(kwargs['y']), 6)
        elif self._element_y in kwargs:
            return round(float(kwargs[self._element_y]), 6)
        elif self._type == 3 and self._element_1my in kwargs:
            return round(1. - float(kwargs[self._element_1my]), 6)
        else:
            raise TypeError()