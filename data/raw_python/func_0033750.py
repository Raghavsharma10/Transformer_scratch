def get_server_public(self, password_verifier, server_private):
        """B = (k*v + g^b) % N

        :param int password_verifier:
        :param int server_private:
        :rtype: int
        """
        return ((self._mult * password_verifier) + pow(self._gen, server_private, self._prime)) % self._prime