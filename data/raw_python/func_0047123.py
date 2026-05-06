def match_description(self, description=None, string_match_type=None, match=None):
        """Adds a description name to match.

        Multiple description matches can be added to perform a boolean
        ``OR`` among them.

        arg:    description (string): description to match
        arg:    string_match_type (osid.type.Type): the string match
                type
        arg:    match (boolean): ``true`` for a positive match,
                ``false`` for a negative match
        raise:  InvalidArgument - ``description`` is not of
                ``string_match_type``
        raise:  NullArgument - ``description`` or ``string_match_type``
                is ``null``
        raise:  Unsupported -
                ``supports_string_match_type(string_match_type)`` is
                ``false``
        *compliance: mandatory -- This method must be implemented.*

        """
        self._match_display_text('description', description, string_match_type, match)