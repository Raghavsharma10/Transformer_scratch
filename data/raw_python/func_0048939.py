def match_keyword(self, keyword, string_match_type=DEFAULT_STRING_MATCH_TYPE, match=True):
        """Adds a keyword to match.

        Multiple keywords can be added to perform a boolean ``OR`` among
        them. A keyword may be applied to any of the elements defined in
        this object such as the display name, description or any method
        defined in an interface implemented by this object.

        arg:    keyword (string): keyword to match
        arg:    string_match_type (osid.type.Type): the string match
                type
        arg:    match (boolean): ``true`` for a positive match,
                ``false`` for a negative match
        raise:  InvalidArgument - ``keyword`` is not of
                ``string_match_type``
        raise:  NullArgument - ``keyword`` or ``string_match_type`` is
                ``null``
        raise:  Unsupported -
                ``supports_string_match_type(string_match_type)`` is
                ``false``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Note: this currently ignores match argument
        match_value = self._get_string_match_value(keyword, string_match_type)
        for field_name in self._keyword_fields:
            if field_name not in self._keyword_terms:
                self._keyword_terms[field_name] = {'$in': list()}
            self._keyword_terms[field_name]['$in'].append(match_value)