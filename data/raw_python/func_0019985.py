def convert(cls, obj, parent):
        """Converts objects to registered tracked types

        This checks the type of the given object against the registered tracked
        types. When a match is found, the given object will be converted to the
        tracked type, its parent set to the provided parent, and returned.

        If its type does not occur in the registered types mapping, the object
        is returned unchanged.
        """
        replacement_type = cls._type_mapping.get(type(obj))
        if replacement_type is not None:
            new = replacement_type(obj)
            new.parent = parent
            return new
        return obj