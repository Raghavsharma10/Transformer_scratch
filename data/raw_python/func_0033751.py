def get_common_password_hash(self, salt):
        """x = H(s | H(I | ":" | P))

        :param int salt:
        :rtype: int
        """
        password = self._password
        if password is None:
            raise SRPException('User password should be in context for this scenario.')

        return self.hash(salt, self.hash(self._user, password, joiner=':'))