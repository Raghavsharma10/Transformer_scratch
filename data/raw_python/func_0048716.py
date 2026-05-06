def _brace_key(self, key):
        """
        key: 'x' -> '{x}'
        """
        if isinstance(key, six.integer_types):
            t = str
            key = t(key)
        else:
            t = type(key)
        return t(u'{') + key + t(u'}')