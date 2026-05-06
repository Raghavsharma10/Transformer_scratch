def getlist(self, name):
        """
        Retrieve given property from class/instance, ensuring it is a list.
        Also determine whether the list contains simple text/numeric values or
        nested dictionaries (a "complex" list)
        """
        value = self.getvalue(name)
        complex = {}

        def str_value(val):
            # TODO: nonlocal complex
            if isinstance(val, dict):
                complex['complex'] = True
                return val
            else:
                return str(val)

        if value is None:
            pass
        else:
            value = [str_value(val) for val in as_list(value)]

        return value, bool(complex)