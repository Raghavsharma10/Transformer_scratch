def update(self, id, **dict):
        '''Update a given item with the passed data.'''
        if not self._item_path:
            raise AttributeError('update is not available for %s' % self._item_name)
        target = (self._update_path or self._item_path) % id
        payload = json.dumps({self._item_type:dict})
        self._redmine.put(target, payload)
        return None