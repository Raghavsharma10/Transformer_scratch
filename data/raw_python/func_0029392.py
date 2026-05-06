def attribute_result(cls, sprites):
        """Return mapping of attributes to if they were initialized or not."""
        retval = dict((x, True) for x in cls.ATTRIBUTES)
        for properties in sprites.values():
            for attribute, state in properties.items():
                retval[attribute] &= state != cls.STATE_MODIFIED
        return retval