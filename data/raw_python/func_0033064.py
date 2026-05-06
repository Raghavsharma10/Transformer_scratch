def unfinished_objects(self):
        '''
        Leaves only versions of those objects that has some version with
        `_end == None` or with `_end > right cutoff`.
        '''
        mask = self._end_isnull
        if self._rbound is not None:
            mask = mask | (self._end > self._rbound)
        oids = set(self[mask]._oid.tolist())
        return self[self._oid.apply(lambda oid: oid in oids)]