def _remove_string_from_commastring(self, field, string):
        # type: (str, str) -> bool
        """Remove a string from a comma separated list of strings

        Args:
            field (str): Field containing comma separated list
            string (str): String to remove

        Returns:
            bool: True if string removed or False if not
        """
        commastring = self.data.get(field, '')
        if string in commastring:
            self.data[field] = commastring.replace(string, '')
            return True
        return False