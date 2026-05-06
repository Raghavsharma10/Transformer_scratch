def _has_y(self, kwargs):
        '''Returns True if y is explicitly defined in kwargs'''
        return (('y' in kwargs) or (self._element_y in kwargs) or
                (self._type == 3 and self._element_1my in kwargs))