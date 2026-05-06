def iteritems(self, **options):
        '''Return a query interator with (id, object) pairs.'''
        iter = self.query(**options)
        while True:
            obj = iter.next()
            yield (obj.id, obj)