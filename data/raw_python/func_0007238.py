def add(self, item):
        '''Add *item*.'''
        if not item in self:
            index = bisect.bisect_right(self._members, item)
            self._members.insert(index, item)