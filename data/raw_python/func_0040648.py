def from_pubkey(cls: Type[CRCPubkeyType], pubkey: str) -> CRCPubkeyType:
        """
        Return CRCPubkey instance from public key string

        :param pubkey: Public key
        :return:
        """
        hash_root = hashlib.sha256()
        hash_root.update(base58.b58decode(pubkey))
        hash_squared = hashlib.sha256()
        hash_squared.update(hash_root.digest())
        b58_checksum = ensure_str(base58.b58encode(hash_squared.digest()))

        crc = b58_checksum[:3]
        return cls(pubkey, crc)