def fields(self):
        '''Return a tuple of ordered fields for this :class:`ColumnTS`.'''
        key = self.id + ':fields'
        encoding = self.client.encoding
        return tuple(sorted((f.decode(encoding)
                             for f in self.client.smembers(key))))