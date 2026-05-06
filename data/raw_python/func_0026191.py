def _handle_value(self, value):
        """
        Given a value string, unquote, remove comment,
        handle lists. (including empty and single member lists)
        """
        if self._inspec:
            # Parsing a configspec so don't handle comments
            return (value, '')
        # do we look for lists in values ?
        if not self.list_values:
            mat = self._nolistvalue.match(value)
            if mat is None:
                raise SyntaxError()
            # NOTE: we don't unquote here
            return mat.groups()
        #
        mat = self._valueexp.match(value)
        if mat is None:
            # the value is badly constructed, probably badly quoted,
            # or an invalid list
            raise SyntaxError()
        (list_values, single, empty_list, comment) = mat.groups()
        if (list_values == '') and (single is None):
            # change this if you want to accept empty values
            raise SyntaxError()
        # NOTE: note there is no error handling from here if the regex
        # is wrong: then incorrect values will slip through
        if empty_list is not None:
            # the single comma - meaning an empty list
            return ([], comment)
        if single is not None:
            # handle empty values
            if list_values and not single:
                # FIXME: the '' is a workaround because our regex now matches
                #   '' at the end of a list if it has a trailing comma
                single = None
            else:
                single = single or '""'
                single = self._unquote(single)
        if list_values == '':
            # not a list value
            return (single, comment)
        the_list = self._listvalueexp.findall(list_values)
        the_list = [self._unquote(val) for val in the_list]
        if single is not None:
            the_list += [single]
        return (the_list, comment)