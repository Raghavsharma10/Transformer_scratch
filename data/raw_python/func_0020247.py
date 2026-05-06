def ipop(self, index):
        '''Pop a value at *index* from the :class:`TS`. Return ``None`` if
index is not out of bound.'''
        backend = self.backend
        res = backend.structure(self).ipop(index)
        return backend.execute(res,
                               lambda r: self._load_get_data(r, index, None))