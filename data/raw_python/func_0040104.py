def from_inline(cls: Type[IdentityType], version: int, currency: str, inline: str) -> IdentityType:
        """
        Return Identity instance from inline Identity string
        :param version: Document version number
        :param currency: Name of the currency
        :param inline: Inline string of the Identity
        :return:
        """
        selfcert_data = Identity.re_inline.match(inline)
        if selfcert_data is None:
            raise MalformedDocumentError("Inline self certification")
        pubkey = selfcert_data.group(1)
        signature = selfcert_data.group(2)
        ts = BlockUID.from_str(selfcert_data.group(3))
        uid = selfcert_data.group(4)

        return cls(version, currency, pubkey, uid, ts, signature)