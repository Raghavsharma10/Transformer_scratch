def remove(self, function, update=True):
        """ Remove a function from the chain.
        """
        self._funcs.remove(function)
        self._remove_dep(function)
        if update:
            self._update()