def pop(self):
        """Leave the current scope
        :returns: TODO

        """
        res = self._scope_stack.pop()
        self._dlog("popping scope, scope level = {}".format(self.level()))
        self._curr_scope = self._scope_stack[-1]
        return res