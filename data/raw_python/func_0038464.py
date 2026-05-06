def _yield_all(self, l):
        '''
        Given a iterable like list or tuple the function yields each of its
        items with _yield
        '''
        if l is not None:
            if type(l) in [list, tuple]:
                for f in l:
                    for x in self._yield(f): yield x
            else:
                for x in self._yield(l): yield x