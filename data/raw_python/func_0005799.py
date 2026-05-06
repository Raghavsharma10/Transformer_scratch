def header_collect(self, name, target_var, pull=False):
        """Store the specified response header in a request var
        (optionally removing it from the response).

        :param str|unicode name:

        :param str|unicode target_var:

        :param bool pull: Whether to remove header from response.

        """
        self._set(
            'pull-header' if pull else 'collect-header',
            '%s %s' % (name, target_var), multi=True)

        return self._section