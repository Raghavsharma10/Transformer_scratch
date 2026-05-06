def encrypt(self, pubkey: str, nonce: Union[str, bytes], text: Union[str, bytes]) -> str:
        """
        Encrypt message text with the public key of the recipient and a nonce

        The nonce must be a 24 character string (you can use libnacl.utils.rand_nonce() to get one)
        and unique for each encrypted message.

        Return base58 encoded encrypted message

        :param pubkey: Base58 encoded public key of the recipient
        :param nonce: Unique nonce
        :param text: Message to encrypt
        :return:
        """
        text_bytes = ensure_bytes(text)
        nonce_bytes = ensure_bytes(nonce)
        recipient_pubkey = PublicKey(pubkey)
        crypt_bytes = libnacl.public.Box(self, recipient_pubkey).encrypt(text_bytes, nonce_bytes)
        return Base58Encoder.encode(crypt_bytes[24:])