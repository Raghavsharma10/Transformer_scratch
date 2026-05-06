def from_inline(cls: Type[MembershipType], version: int, currency: str, membership_type: str,
                    inline: str) -> MembershipType:
        """
        Return Membership instance from inline format

        :param version: Version of the document
        :param currency: Name of the currency
        :param membership_type: "IN" or "OUT" to enter or exit membership
        :param inline: Inline string format
        :return:
        """
        data = Membership.re_inline.match(inline)
        if data is None:
            raise MalformedDocumentError("Inline membership ({0})".format(inline))
        issuer = data.group(1)
        signature = data.group(2)
        membership_ts = BlockUID.from_str(data.group(3))
        identity_ts = BlockUID.from_str(data.group(4))
        uid = data.group(5)
        return cls(version, currency, issuer, membership_ts, membership_type, uid, identity_ts, signature)