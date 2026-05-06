def _index(self, item):
        '''Return index of *item* in member list or -1 if not present.'''
        index = bisect.bisect_left(self._members, item)
        if index != len(self) and self._members[index] == item:
            return index

        return -1