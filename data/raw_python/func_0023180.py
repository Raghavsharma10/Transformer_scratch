def insert(self, index, function, update=True):
        """ Insert a new function into the chain at *index*.
        """
        self._funcs.insert(index, function)
        self._add_dep(function)
        if update:
            self._update()