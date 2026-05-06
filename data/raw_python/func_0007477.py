def fetch_by_code(self, code):
        """
        Returns an AuthorizationCode.

        :param code: The authorization code.
        :return: An instance of :class:`oauth2.datatype.AuthorizationCode`.
        :raises: :class:`AuthCodeNotFound` if no data could be retrieved for
                 given code.

        """
        if code not in self.auth_codes:
            raise AuthCodeNotFound

        return self.auth_codes[code]