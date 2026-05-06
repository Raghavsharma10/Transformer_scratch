def bytes_array(self):
        '''Get the param as an array of raw byte strings.'''
        assert len(self.dimensions) == 2, \
            '{}: cannot get value as bytes array!'.format(self.name)
        l, n = self.dimensions
        return [self.bytes[i*l:(i+1)*l] for i in range(n)]