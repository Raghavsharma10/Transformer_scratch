def add_local(self, field_name, field):
        """Add a local variable in the current scope

        :field_name: The field's name
        :field: The field
        :returns: None

        """
        self._dlog("adding local '{}'".format(field_name))
        field._pfp__name = field_name
        # TODO do we allow clobbering of locals???
        self._curr_scope["vars"][field_name] = field