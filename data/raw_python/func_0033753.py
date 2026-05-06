def get_common_session_key_proof_hash(self, session_key, session_key_proof, client_public):
        """H(A | M | K)

        :param bytes session_key:
        :param bytes session_key_proof:
        :param int client_public:
        :rtype: bytes
        """
        return self.hash(client_public, session_key_proof, session_key, as_bytes=True)