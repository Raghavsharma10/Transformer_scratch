def pop_back(self):
        '''Remove the last element from the :class:`Sequence`.'''
        backend = self.backend
        return backend.execute(backend.structure(self).pop_back(),
                               self.value_pickler.loads)