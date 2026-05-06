def find_whitespace_pattern(self):
        """
        Try to find a whitespace pattern in the existing parameters
        to be applied to a newly added parameter
        """
        name_ws = []
        value_ws = []
        for entry in self._entries:
            name_ws.append(get_whitespace(entry.name))
            if entry.value != '':
                value_ws.append(get_whitespace(entry._value))  # _value is unstripped
        if len(value_ws) >= 1:
            value_ws = most_common(value_ws)
        else:
            value_ws = ('', ' ')
        if len(name_ws) >= 1:
            name_ws = most_common(name_ws)
        else:
            name_ws = (' ', '')
        return name_ws, value_ws