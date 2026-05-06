def _as_array(self, fmt):
        '''Unpack the raw bytes of this param using the given data format.'''
        assert self.dimensions, \
            '{}: cannot get value as {} array!'.format(self.name, fmt)
        elems = array.array(fmt)
        elems.fromstring(self.bytes)
        return np.array(elems).reshape(self.dimensions)