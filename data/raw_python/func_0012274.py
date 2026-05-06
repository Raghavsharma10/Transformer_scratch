def set_values(self, values, separator='\n', indent=4*' '):
        """Sets the value to a given list of options, e.g. multi-line values

        Args:
            values (list): list of values
            separator (str): separator for values, default: line separator
            indent (str): indentation depth in case of line separator
        """
        self._updated = True
        self._multiline_value_joined = True
        self._values = values
        if separator == '\n':
            values.insert(0, '')
            separator = separator + indent
        self._value = separator.join(values)