def createx(self, object_type, under=None, attributes=None, **kwattrs):
        """Create a new automation object.

        Arguments:
        object_type -- Type of object to create.
        under       -- Handle of the parent of the new object.
        attributes  -- Dictionary of attributes (name-value pairs).
        kwattrs     -- Optional keyword attributes (name=value pairs).

        Return:
        Dictionary containing handle of newly created object.

        """
        self._check_session()
        params = {'object_type': object_type}
        if under:
            params['under'] = under
        if attributes:
            params.update(attributes)
        if kwattrs:
            params.update(kwattrs)

        status, data = self._rest.post_request('objects', None, params)
        return data