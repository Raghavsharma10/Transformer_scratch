def string_array(self):
        '''Get the param as a array of unicode strings.'''
        assert len(self.dimensions) == 2, \
            '{}: cannot get value as string array!'.format(self.name)
        l, n = self.dimensions
        return [self.bytes[i*l:(i+1)*l].decode('utf-8') for i in range(n)]