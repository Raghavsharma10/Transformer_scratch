def _add_strings_to_commastring(self, field, strings):
        # type: (str, List[str]) -> bool
        """Add a list of strings to a comma separated list of strings

        Args:
            field (str): Field containing comma separated list
            strings (List[str]): list of strings to add

        Returns:
            bool: True if all strings added or False if any already present.
        """
        allstringsadded = True
        for string in strings:
            if not self._add_string_to_commastring(field, string):
                allstringsadded = False
        return allstringsadded