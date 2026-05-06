def pop_prefix(string: str):
        """Erases the prefix and returns it.
        :throws IndexError: There is no prefix.
        :return A set with two elements: 1- the prefix, 2- the type without it.
        """
        result = string.split(Naming.TYPE_PREFIX)
        if len(result) == 1:
            result = string.split(Naming.RESOURCE_PREFIX)
            if len(result) == 1:
                raise IndexError()
        return result