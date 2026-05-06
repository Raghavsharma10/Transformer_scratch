def _has_z(self, kwargs):
        '''
        Returns True if type is 1 or 2 and z is explicitly defined in kwargs.
        '''
        return ((self._type == 1 or self._type ==2) and
                (('z' in kwargs) or (self._element_z in kwargs)))