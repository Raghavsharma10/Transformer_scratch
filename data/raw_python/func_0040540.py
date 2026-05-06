def from_private_key(path: str) -> SigningKeyType:
        """
        Read authentication file
        Add public key attribute

        :param path: Authentication file path
        """
        key = load_key(path)
        key.pubkey = Base58Encoder.encode(key.vk)
        return key