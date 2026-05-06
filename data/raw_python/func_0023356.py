def wrapping(self):
        """ Texture wrapping mode """
        value = self._wrapping
        return value[0] if all([v == value[0] for v in value]) else value