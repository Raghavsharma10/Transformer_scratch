def is_valid(self) -> bool:
        """
        Return True if CRC is valid
        :return:
        """
        return CRCPubkey.from_pubkey(self.pubkey).crc == self.crc