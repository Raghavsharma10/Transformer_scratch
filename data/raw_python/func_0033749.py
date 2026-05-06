def get_server_premaster_secret(self, password_verifier, server_private, client_public, common_secret):
        """S = (A * v^u) ^ b % N

        :param int password_verifier:
        :param int server_private:
        :param int client_public:
        :param int common_secret:
        :rtype: int
        """
        return pow((client_public * pow(password_verifier, common_secret, self._prime)), server_private, self._prime)