def from_str(cls: Type[CRCPubkeyType], crc_pubkey: str) -> CRCPubkeyType:
        """
        Return CRCPubkey instance from CRC public key string

        :param crc_pubkey: CRC public key
        :return:
        """
        data = CRCPubkey.re_crc_pubkey.match(crc_pubkey)
        if data is None:
            raise Exception("Could not parse CRC public key {0}".format(crc_pubkey))
        pubkey = data.group(1)
        crc = data.group(2)
        return cls(pubkey, crc)