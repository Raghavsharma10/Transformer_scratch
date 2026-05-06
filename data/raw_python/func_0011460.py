def add_type(self, new_name, orig_names):
        """Record the typedefd name for orig_names. Resolve orig_names
        to their core names and save those.

        :new_name: TODO
        :orig_names: TODO
        :returns: TODO

        """
        self._dlog("adding a type '{}'".format(new_name))
        # TODO do we allow clobbering of types???
        res = copy.copy(orig_names)
        resolved_names = self._resolve_name(res[-1])
        if resolved_names is not None:
            res.pop()
            res += resolved_names

        self._curr_scope["types"][new_name] = res