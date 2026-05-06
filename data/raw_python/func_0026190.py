def _quote(self, value, multiline=True):
        """
        Return a safely quoted version of a value.

        Raise a ConfigObjError if the value cannot be safely quoted.
        If multiline is ``True`` (default) then use triple quotes
        if necessary.

        * Don't quote values that don't need it.
        * Recursively quote members of a list and return a comma joined list.
        * Multiline is ``False`` for lists.
        * Obey list syntax for empty and single member lists.

        If ``list_values=False`` then the value is only quoted if it contains
        a ``\\n`` (is multiline) or '#'.

        If ``write_empty_values`` is set, and the value is an empty string, it
        won't be quoted.
        """
        if multiline and self.write_empty_values and value == '':
            # Only if multiline is set, so that it is used for values not
            # keys, and not values that are part of a list
            return ''

        if multiline and isinstance(value, (list, tuple)):
            if not value:
                return ','
            elif len(value) == 1:
                return self._quote(value[0], multiline=False) + ','
            return ', '.join([self._quote(val, multiline=False)
                for val in value])
        if not isinstance(value, string_types):
            if self.stringify:
                value = str(value)
            else:
                raise TypeError('Value "%s" is not a string.' % value)

        if not value:
            return '""'

        no_lists_no_quotes = not self.list_values and '\n' not in value and '#' not in value
        need_triple = multiline and ((("'" in value) and ('"' in value)) or ('\n' in value ))
        hash_triple_quote = multiline and not need_triple and ("'" in value) and ('"' in value) and ('#' in value)
        check_for_single = (no_lists_no_quotes or not need_triple) and not hash_triple_quote

        if check_for_single:
            if not self.list_values:
                # we don't quote if ``list_values=False``
                quot = noquot
            # for normal values either single or double quotes will do
            elif '\n' in value:
                # will only happen if multiline is off - e.g. '\n' in key
                raise ConfigObjError('Value "%s" cannot be safely quoted.' % value)
            elif ((value[0] not in wspace_plus) and
                    (value[-1] not in wspace_plus) and
                    (',' not in value)):
                quot = noquot
            else:
                quot = self._get_single_quote(value)
        else:
            # if value has '\n' or "'" *and* '"', it will need triple quotes
            quot = self._get_triple_quote(value)

        if quot == noquot and '#' in value and self.list_values:
            quot = self._get_single_quote(value)

        return quot % value