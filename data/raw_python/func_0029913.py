def increment(cls, v):
        """Increment the version number of an object number of object number string"""
        if not isinstance(v, ObjectNumber):
            v = ObjectNumber.parse(v)

        return v.rev(v.revision+1)