def create_property(name, ptype):
        """
        Creates a custom property with a getter that performs computing
        functionality (if available) and raise a type error if setting
        with the wrong type.

        Note:
            By default, the setter attempts to convert the object to the
            correct type; a type error is raised if this fails.
        """
        pname = '_' + name
        def getter(self):
            # This will be where the data is store (e.g. self._name)
            # This is the default property "getter" for container data objects.
            # If the property value is None, this function will check for a
            # convenience method with the signature, self.compute_name() and call
            # it prior to returning the property value.
            if not hasattr(self, pname) and hasattr(self, '{}{}'.format(self._getter_prefix, pname)):
                self['{}{}'.format(self._getter_prefix, pname)]()
            if not hasattr(self, pname):
                raise AttributeError('Please compute or set {} first.'.format(name))
            return getattr(self, pname)

        def setter(self, obj):
            # This is the default property "setter" for container data objects.
            # Prior to setting a property value, this function checks that the
            # object's type is correct.
            if not isinstance(obj, ptype):
                try:
                    obj = ptype(obj)
                except Exception:
                    raise TypeError('Must be able to convert object {0} to {1} (or must be of type {1})'.format(name, ptype))
            setattr(self, pname, obj)

        def deleter(self):
            # Deletes the property's value.
            del self[pname]

        return property(getter, setter, deleter)