def from_signed_raw(cls: Type[MembershipType], signed_raw: str) -> MembershipType:
        """
        Return Membership instance from signed raw format

        :param signed_raw: Signed raw format string
        :return:
        """
        lines = signed_raw.splitlines(True)
        n = 0

        version = int(Membership.parse_field("Version", lines[n]))
        n += 1

        Membership.parse_field("Type", lines[n])
        n += 1

        currency = Membership.parse_field("Currency", lines[n])
        n += 1

        issuer = Membership.parse_field("Issuer", lines[n])
        n += 1

        membership_ts = BlockUID.from_str(Membership.parse_field("Block", lines[n]))
        n += 1

        membership_type = Membership.parse_field("Membership", lines[n])
        n += 1

        uid = Membership.parse_field("UserID", lines[n])
        n += 1

        identity_ts = BlockUID.from_str(Membership.parse_field("CertTS", lines[n]))
        n += 1

        signature = Membership.parse_field("Signature", lines[n])
        n += 1

        return cls(version, currency, issuer, membership_ts,
                   membership_type, uid, identity_ts, signature)