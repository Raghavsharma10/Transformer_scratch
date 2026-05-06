def from_inline(cls: Type[RevocationType], version: int, currency: str, inline: str) -> RevocationType:
        """
        Return Revocation document instance from inline string

        Only self.pubkey is populated.
        You must populate self.identity with an Identity instance to use raw/sign/signed_raw methods

        :param version: Version number
        :param currency: Name of the currency
        :param inline: Inline document

        :return:
        """
        cert_data = Revocation.re_inline.match(inline)
        if cert_data is None:
            raise MalformedDocumentError("Revokation")
        pubkey = cert_data.group(1)
        signature = cert_data.group(2)
        return cls(version, currency, pubkey, signature)