def _register(self, name, AttrCls):
        """Register a new attribute to take care of with dump and load

           Arguments:
            | ``name``  --  the name to be used in the dump file
            | ``AttrCls``  --  an attr class describing the attribute
        """
        if not issubclass(AttrCls, StateAttr):
            raise TypeError("The second argument must a StateAttr instance.")
        if len(name) > 40:
            raise ValueError("Name can count at most 40 characters.")
        self._fields[name] = AttrCls(self._owner, name)