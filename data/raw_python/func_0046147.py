def match_display_names(self, value, match):
        """stub"""
        if value is None:
            raise NullArgument('value must not be None')
        if not utilities.is_string(value):
            raise InvalidArgument('value must be a string')
        if match is None:
            raise NullArgument('match must not be None')
        if not isinstance(match, bool):
            raise InvalidArgument('match must be a bool')
        self._my_osid_query._add_match('displayNames.text', str(value).lower(), match)