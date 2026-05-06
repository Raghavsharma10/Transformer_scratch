def config(self, handle, attributes=None, **kwattrs):
        """Sets or modifies one or more object attributes or relations.

        Arguments can be supplied either as a dictionary or as keyword
        arguments.  Examples:
            stc.config('port1', location='//10.1.2.3/1/1')
            stc.config('port2', {'location': '//10.1.2.3/1/2'})

        Arguments:
        handle     -- Handle of object to modify.
        attributes -- Dictionary of attributes (name-value pairs).
        kwattrs    -- Optional keyword attributes (name=value pairs).

        """
        self._check_session()
        if kwattrs:
            if attributes:
                attributes.update(kwattrs)
            else:
                attributes = kwattrs
        self._rest.put_request('objects', str(handle), attributes)