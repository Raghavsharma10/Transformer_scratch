def inline(self) -> str:
        """
        Return an inline string of the Identity
        :return:
        """
        return "{pubkey}:{signature}:{timestamp}:{uid}".format(
            pubkey=self.pubkey,
            signature=self.signatures[0],
            timestamp=self.timestamp,
            uid=self.uid)