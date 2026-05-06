def match_display_name(self, display_name=None, string_match_type=None, match=None):
        """Adds a display name to match.

        Multiple display name matches can be added to perform a boolean
        ``OR`` among them.

        arg:    display_name (string): display name to match
        arg:    string_match_type (osid.type.Type): the string match
                type
        arg:    match (boolean): ``true`` for a positive match,
                ``false`` for a negative match
        raise:  InvalidArgument - ``display_name`` is not of
                ``string_match_type``
        raise:  NullArgument - ``display_name`` or ``string_match_type``
                is ``null``
        raise:  Unsupported -
                ``supports_string_match_type(string_match_type)`` is
                ``false``
        *compliance: mandatory -- This method must be implemented.*

        """
        self._match_display_text('displayName', display_name, string_match_type, match)