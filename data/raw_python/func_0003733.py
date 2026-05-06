def set(self, new_fields, subset=None):
        """Assign the registered fields based on a dictionary

           Argument:
            | ``new_fields``  --  the dictionary with the data to be assigned to
                                  the attributes

           Optional argument:
            | ``subset``  --  a list of names to restrict the fields that are
                              effectively overwritten
        """
        for name in new_fields:
            if name not in self._fields and (subset is None or name in subset):
                raise ValueError("new_fields contains an unknown field '%s'." % name)
        if subset is not None:
            for name in subset:
                if name not in self._fields:
                    raise ValueError("name '%s' in subset is not a known field in self._fields." % name)
                if name not in new_fields:
                    raise ValueError("name '%s' in subset is not a known field in new_fields." % name)
        if subset is None:
            if len(new_fields) != len(self._fields):
                raise ValueError("new_fields contains too many fields.")
        for name, attr in self._fields.items():
            if name in subset:
                attr.set(new_fields[name])