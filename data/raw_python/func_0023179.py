def append(self, function, update=True):
        """ Append a new function to the end of this chain.
        """
        self._funcs.append(function)
        self._add_dep(function)
        if update:
            self._update()