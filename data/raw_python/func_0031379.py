def put(self, item):
        ''' store item in sqlite database
        '''
        if isinstance(item, self._item_class):
            self._put_one(item)
        elif isinstance(item, (list, tuple)):
            self._put_many(item)
        else:
            raise RuntimeError('Unknown item(s) type, %s' % type(item))