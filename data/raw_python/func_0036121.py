def pop(self, key, *args, **kwargs):
        """Remove specified key and return the corresponding value.

        :keyword default: If key is not found, ``default`` is returned if given,
            otherwise :exc:`KeyError` is raised.

        """
        try:
            val = self[key]
        except KeyError:
            if len(args):
                return args[0]
            if "default" in kwargs:
                return kwargs["default"]
            raise

        try:
            del(self[key])
        except KeyError:
            pass

        return val