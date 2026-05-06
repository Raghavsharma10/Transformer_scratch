def delete(self, id):
        '''Delete a single item with the given ID'''
        if not self._item_path:
            raise AttributeError('delete is not available for %s' % self._item_name)
        target = self._item_path % id
        self._redmine.delete(target)
        return None