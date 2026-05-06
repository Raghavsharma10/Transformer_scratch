def remove(self, *keys):
        '''Remove *keys* from the key-value container.'''
        dumps = self.pickler.dumps
        self.cache.remove([dumps(v) for v in keys])