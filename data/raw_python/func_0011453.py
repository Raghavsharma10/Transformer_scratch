def clone(self):
        """Return a new Scope object that has the curr_scope
        pinned at the current one
        :returns: A new scope object
        """
        self._dlog("cloning the stack")
        # TODO is this really necessary to create a brand new one?
        # I think it is... need to think about it more.
        # or... are we going to need ref counters and a global
        # scope object that allows a view into (or a snapshot of)
        # a specific scope stack?
        res = Scope(self._log)
        res._scope_stack = self._scope_stack
        res._curr_scope = self._curr_scope
        return res