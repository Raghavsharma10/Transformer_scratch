def merge(self, *args):
        """
        Merge multiple dictionary objects into one.

        :param variadic args: Multiple dictionary items

        :return dict
        """
        values = []

        for entry in args:
            values = values + list(entry.items())

        return dict(values)