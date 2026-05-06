def from_signed_raw(cls: Type[RevocationType], signed_raw: str) -> RevocationType:
        """
        Return Revocation document instance from a signed raw string

        :param signed_raw: raw document file in duniter format
        :return:
        """
        lines = signed_raw.splitlines(True)
        n = 0

        version = int(Revocation.parse_field("Version", lines[n]))
        n += 1

        Revocation.parse_field("Type", lines[n])
        n += 1

        currency = Revocation.parse_field("Currency", lines[n])
        n += 1

        issuer = Revocation.parse_field("Issuer", lines[n])
        n += 1

        identity_uid = Revocation.parse_field("IdtyUniqueID", lines[n])
        n += 1

        identity_timestamp = Revocation.parse_field("IdtyTimestamp", lines[n])
        n += 1

        identity_signature = Revocation.parse_field("IdtySignature", lines[n])
        n += 1

        signature = Revocation.parse_field("Signature", lines[n])
        n += 1

        identity = Identity(version, currency, issuer, identity_uid, identity_timestamp, identity_signature)

        return cls(version, currency, identity, signature)