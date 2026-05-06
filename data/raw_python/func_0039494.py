def from_inline(cls: Type[CertificationType], version: int, currency: str, blockhash: Optional[str],
                    inline: str) -> CertificationType:
        """
        Return Certification instance from inline document

        Only self.pubkey_to is populated.
        You must populate self.identity with an Identity instance to use raw/sign/signed_raw methods

        :param version: Version of document
        :param currency: Name of the currency
        :param blockhash: Hash of the block
        :param inline: Inline document
        :return:
        """
        cert_data = Certification.re_inline.match(inline)
        if cert_data is None:
            raise MalformedDocumentError("Certification ({0})".format(inline))
        pubkey_from = cert_data.group(1)
        pubkey_to = cert_data.group(2)
        blockid = int(cert_data.group(3))
        if blockid == 0 or blockhash is None:
            timestamp = BlockUID.empty()
        else:
            timestamp = BlockUID(blockid, blockhash)

        signature = cert_data.group(4)
        return cls(version, currency, pubkey_from, pubkey_to, timestamp, signature)