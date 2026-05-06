def flush(self, meta=None):
        '''Flush all model keys from the database'''
        pattern = self.basekey(meta) if meta else self.namespace
        return self.client.delpattern('%s*' % pattern)