def get_common_session_key_proof(self, session_key, salt, server_public, client_public):
        """M = H(H(N) XOR H(g) | H(U) | s | A | B | K)

        :param bytes session_key:
        :param int salt:
        :param int server_public:
        :param int client_public:
        :rtype: bytes
        """
        h = self.hash
        prove = h(
            h(self._prime) ^ h(self._gen),
            h(self._user),
            salt,
            client_public,
            server_public,
            session_key,
            as_bytes=True
        )
        return prove