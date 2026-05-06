def header_remove(self, value):
        """Automatically remove specified HTTP header from the response.

        :param str|unicode value:

        """
        self._set('del-header', value, multi=True)

        return self._section