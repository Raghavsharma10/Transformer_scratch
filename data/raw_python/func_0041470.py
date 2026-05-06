def encrypt_seal(self, data: Union[str, bytes]) -> bytes:
        """
        Encrypt data with a curve25519 version of the ed25519 public key

        :param data: Bytes data to encrypt
        """
        curve25519_public_key = libnacl.crypto_sign_ed25519_pk_to_curve25519(self.pk)
        return libnacl.crypto_box_seal(ensure_bytes(data), curve25519_public_key)