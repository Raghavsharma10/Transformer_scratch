def raw(self) -> str:
        """
        Return Revocation raw document string

        :return:
        """
        if not isinstance(self.identity, Identity):
            raise MalformedDocumentError("Can not return full revocation document created from inline")

        return """Version: {version}
Type: Revocation
Currency: {currency}
Issuer: {pubkey}
IdtyUniqueID: {uid}
IdtyTimestamp: {timestamp}
IdtySignature: {signature}
""".format(version=self.version,
           currency=self.currency,
           pubkey=self.identity.pubkey,
           uid=self.identity.uid,
           timestamp=self.identity.timestamp,
           signature=self.identity.signatures[0])