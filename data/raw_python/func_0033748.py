def get_client_premaster_secret(self, password_hash, server_public, client_private, common_secret):
        """S = (B - (k * g^x)) ^ (a + (u * x)) % N

        :param int server_public:
        :param int password_hash:
        :param int client_private:
        :param int common_secret:
        :rtype: int
        """
        password_verifier = self.get_common_password_verifier(password_hash)
        return pow(
            (server_public - (self._mult * password_verifier)),
            (client_private + (common_secret * password_hash)), self._prime)