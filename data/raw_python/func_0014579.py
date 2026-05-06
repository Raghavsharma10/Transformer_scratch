def create(self, object_type, under=None, attributes=None, **kwattrs):
        """Create a new automation object.

        Arguments:
        object_type -- Type of object to create.
        under       -- Handle of the parent of the new object.
        attributes  -- Dictionary of attributes (name-value pairs).
        kwattrs     -- Optional keyword attributes (name=value pairs).

        Return:
        Handle of newly created object.

        """
        data = self.createx(object_type, under, attributes, **kwattrs)
        return data['handle']