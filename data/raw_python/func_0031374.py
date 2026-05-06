def get(self, criteria=None, offset=None, limit=None):
        ''' returns items selected by criteria

        If the criteria is not defined, get() returns all items.
        '''
        if criteria is None and limit is None:
            return self._get_all()
        elif limit is not None and limit == 1:
            return self.get_one(criteria)
        else:
            return self._get_with_criteria(criteria, offset=offset, limit=limit)