def discard(self, item):
        '''Remove *item*.'''
        index = self._index(item)
        if index >= 0:
            del self._members[index]