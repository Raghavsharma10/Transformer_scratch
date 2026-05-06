def get(self, handle, *args):
        """Returns the value(s) of one or more object attributes.

        If multiple arguments, this method returns a dictionary of argument
        names mapped to the value returned by each argument.

        If a single argument is given, then the response is a list of values
        for that argument.

        Arguments:
        handle -- Handle that identifies object to get info for.
        *args  -- Zero or more attributes or relationships.

        Return:
        If multiple input arguments are given:
        {attrib_name:attrib_val, attrib_name:attrib_val, ..}

        If single input argument is given, then a single string value is
        returned.  NOTE: If the string contains multiple substrings, then the
        client will need to parse these.

        """
        self._check_session()
        status, data = self._rest.get_request('objects', str(handle), args)
        return data