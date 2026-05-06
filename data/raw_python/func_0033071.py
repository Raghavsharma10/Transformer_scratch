def _has_x(self, kwargs):
        '''Returns True if x is explicitly defined in kwargs'''
        return (('x' in kwargs) or (self._element_x in kwargs) or
                (self._type == 3 and self._element_1mx in kwargs))