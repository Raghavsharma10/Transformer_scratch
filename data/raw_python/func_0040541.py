def decrypt_seal(self, data: bytes) -> bytes:
        """
        Decrypt bytes data with a curve25519 version of the ed25519 key pair

        :param data: Encrypted data

        :return:
        """
        curve25519_public_key = libnacl.crypto_sign_ed25519_pk_to_curve25519(self.vk)
        curve25519_secret_key = libnacl.crypto_sign_ed25519_sk_to_curve25519(self.sk)
        return libnacl.crypto_box_seal_open(data, curve25519_public_key, curve25519_secret_key)