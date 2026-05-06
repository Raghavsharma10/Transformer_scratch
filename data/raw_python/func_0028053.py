def _add_string_to_commastring(self, field, string):
        # type: (str, str) -> bool
        """Add a string to a comma separated list of strings

        Args:
            field (str): Field containing comma separated list
            string (str): String to add

        Returns:
            bool: True if string added or False if string already present
        """
        if string in self._get_stringlist_from_commastring(field):
            return False
        strings = '%s,%s' % (self.data.get(field, ''), string)
        if strings[0] == ',':
            strings = strings[1:]
        self.data[field] = strings
        return True