def header_add(self, name, value):
        """Automatically add HTTP headers to response.

        :param str|unicode name:

        :param str|unicode value:

        """
        self._set('add-header', '%s: %s' % (name, value), multi=True)

        return self._section