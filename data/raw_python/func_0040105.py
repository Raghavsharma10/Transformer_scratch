def from_signed_raw(cls: Type[IdentityType], signed_raw: str) -> IdentityType:
        """
        Return Identity instance from a signed_raw string
        :param signed_raw: Signed raw document
        :return:
        """
        n = 0
        lines = signed_raw.splitlines(True)

        version = int(Identity.parse_field("Version", lines[n]))
        n += 1

        Identity.parse_field("Type", lines[n])
        n += 1

        currency = Identity.parse_field("Currency", lines[n])
        n += 1

        pubkey = Identity.parse_field("Issuer", lines[n])
        n += 1

        uid = Identity.parse_field("UniqueID", lines[n])
        n += 1

        ts = BlockUID.from_str(Identity.parse_field("Timestamp", lines[n]))
        n += 1

        signature = Identity.parse_field("Signature", lines[n])

        return cls(version, currency, pubkey, uid, ts, signature)