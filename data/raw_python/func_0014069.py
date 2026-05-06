def items(cls):
        """
        :return: List of tuples consisting of every enum value in the form [('NAME', value), ...]
        :rtype: list
        """
        items = [(value.name, key) for key, value in cls.values.items()]
        return sorted(items, key=lambda x: x[1])