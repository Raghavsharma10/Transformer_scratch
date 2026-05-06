def decrypt(self, pubkey: str, nonce: Union[str, bytes], text: str) -> str:
        """
        Decrypt encrypted message text with recipient public key and the unique nonce used by the sender.

        :param pubkey: Public key of the recipient
        :param nonce: Unique nonce used by the sender
        :param text: Encrypted message
        :return:
        """
        sender_pubkey = PublicKey(pubkey)
        nonce_bytes = ensure_bytes(nonce)
        encrypt_bytes = Base58Encoder.decode(text)
        decrypt_bytes = libnacl.public.Box(self, sender_pubkey).decrypt(encrypt_bytes, nonce_bytes)
        return decrypt_bytes.decode('utf-8')