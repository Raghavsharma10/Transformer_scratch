def raw(self) -> str:
        """
        Return a raw document of the certification
        """
        if not isinstance(self.identity, Identity):
            raise MalformedDocumentError("Can not return full certification document created from inline")

        return """Version: {version}
Type: Certification
Currency: {currency}
Issuer: {issuer}
IdtyIssuer: {certified_pubkey}
IdtyUniqueID: {certified_uid}
IdtyTimestamp: {certified_ts}
IdtySignature: {certified_signature}
CertTimestamp: {timestamp}
""".format(version=self.version,
           currency=self.currency,
           issuer=self.pubkey_from,
           certified_pubkey=self.identity.pubkey,
           certified_uid=self.identity.uid,
           certified_ts=self.identity.timestamp,
           certified_signature=self.identity.signatures[0],
           timestamp=self.timestamp)