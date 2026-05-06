def add_var(self, field_name, field, root=False):
        """Add a var to the current scope (vars are fields that
        parse the input stream)

        :field_name: TODO
        :field: TODO
        :returns: TODO

        """
        self._dlog("adding var '{}' (root={})".format(field_name, root))

        # do both so it's not clobbered by intermediate values of the same name
        if root:
            self._scope_stack[0]["vars"][field_name] = field

        # TODO do we allow clobbering of vars???
        self._curr_scope["vars"][field_name] = field