def deserialize(self, value, flags):
        """
        Deserialized values based on flags or just return it if it is not serialized.

        :param value: Serialized or not value.
        :type value: six.string_types, int
        :param flags: Value flags
        :type flags: int
        :return: Deserialized value
        :rtype: six.string_types|int
        """
        FLAGS = self.FLAGS

        if flags & FLAGS['compressed']:  # pragma: no branch
            value = self.compression.decompress(value)

        if flags & FLAGS['binary']:
            return value

        if flags & FLAGS['integer']:
            return int(value)
        elif flags & FLAGS['long']:
            return long(value)
        elif flags & FLAGS['object']:
            buf = BytesIO(value)
            unpickler = self.unpickler(buf)
            return unpickler.load()

        if six.PY3:
            return value.decode('utf8')

        # In Python 2, mimic the behavior of the json library: return a str
        # unless the value contains unicode characters.
        # in Python 2, if value is a binary (e.g struct.pack("<Q") then decode will fail
        try:
            value.decode('ascii')
        except UnicodeDecodeError:
            try:
                return value.decode('utf8')
            except UnicodeDecodeError:
                return value
        else:
            return value