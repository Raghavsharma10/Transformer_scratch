def gecos(self):
        """Force double quoted gecos.

        returns:
            str: The double quoted gecos.
        """
        if not self._gecos:
            return None
        if self._gecos.startswith(text_type('\'')) and self._gecos.endswith(text_type('\'')):
            self._gecos = '\"{0}\"'.format(self._gecos[1:-1])
            return self._gecos
        elif self._gecos.startswith(text_type('\"')) and self._gecos.endswith(text_type('\"')):
            return self._gecos
        else:
            return '\"{0}\"'.format(self._gecos)