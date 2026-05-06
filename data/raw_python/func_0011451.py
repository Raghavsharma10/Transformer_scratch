def level(self):
        """Return the current scope level
        """
        res = len(self._scope_stack)
        if self._parent is not None:
            res += self._parent.level()
        return res