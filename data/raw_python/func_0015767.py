def to_python(self, obj):
        """Converts strings in a data structure to Python types

        It converts datetime-ish things to Python datetimes.

        Override if you want something different.

        :arg obj: Python datastructure

        :returns: Python datastructure with strings converted to
            Python types

        .. Note::

           This does the conversion in-place!

        """
        if isinstance(obj, string_types):
            if len(obj) == 26:
                try:
                    return datetime.strptime(obj, '%Y-%m-%dT%H:%M:%S.%f')
                except (TypeError, ValueError):
                    pass
            elif len(obj) == 19:
                try:
                    return datetime.strptime(obj, '%Y-%m-%dT%H:%M:%S')
                except (TypeError, ValueError):
                    pass
            elif len(obj) == 10:
                try:
                    return datetime.strptime(obj, '%Y-%m-%d')
                except (TypeError, ValueError):
                    pass

        elif isinstance(obj, dict):
            for key, val in obj.items():
                obj[key] = self.to_python(val)

        elif isinstance(obj, list):
            return [self.to_python(item) for item in obj]

        return obj