def pop_front(self):
        '''Remove the first element from of the list.'''
        backend = self.backend
        return backend.execute(backend.structure(self).pop_front(),
                               self.value_pickler.loads)