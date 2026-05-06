def remove(self, item):
        '''Remove ``item`` for the :class:`zset` it it exists.
If found it returns the score of the item removed.'''
        score = self._dict.pop(item, None)
        if score is not None:
            self._sl.remove(score)
            return score