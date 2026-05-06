def raw(self) -> str:
        """
        Return a raw document of the Identity
        :return:
        """
        return """Version: {version}
Type: Identity
Currency: {currency}
Issuer: {pubkey}
UniqueID: {uid}
Timestamp: {timestamp}
""".format(version=self.version,
           currency=self.currency,
           pubkey=self.pubkey,
           uid=self.uid,
           timestamp=self.timestamp)