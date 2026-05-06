def get_common_secret(self, server_public, client_public):
        """u = H(PAD(A) | PAD(B))

        :param int server_public:
        :param int client_public:
        :rtype: int
        """
        return self.hash(self.pad(client_public), self.pad(server_public))