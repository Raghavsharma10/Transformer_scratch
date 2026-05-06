def push(self, new_scope=None):
        """Create a new scope
        :returns: TODO

        """
        if new_scope is None:
            new_scope = {
                "types": {},
                "vars": {}
            }
        self._curr_scope = new_scope
        self._dlog("pushing new scope, scope level = {}".format(self.level()))
        self._scope_stack.append(self._curr_scope)