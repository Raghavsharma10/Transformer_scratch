def _put_many(self, items):
        ''' store items in sqlite database
        '''
        for item in items:
            if not isinstance(item, self._item_class):
                raise RuntimeError('Items mismatch for %s and %s' % (self._item_class, type(item)))
            self._put_one(item)