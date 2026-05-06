def unserialize(cls, string):
        """Unserializes from a string.

        :param string: A string created by :meth:`serialize`.
        """
        id_s, created_s = string.split('_')
        return cls(int(id_s, 16),
                   datetime.utcfromtimestamp(int(created_s, 16)))