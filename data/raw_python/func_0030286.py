def dict(self):
        """A dict that holds key/values for all of the properties in the
        object.

        :return:

        """
        from collections import OrderedDict
        SKIP_KEYS = ()
        return OrderedDict(
            (p.key, getattr(self, p.key)) for p in self.__mapper__.attrs if p.key not in SKIP_KEYS)