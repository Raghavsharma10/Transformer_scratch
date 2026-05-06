def from_signed_raw(cls: Type[CertificationType], signed_raw: str) -> CertificationType:
        """
        Return Certification instance from signed raw document

        :param signed_raw: Signed raw document
        :return:
        """
        n = 0
        lines = signed_raw.splitlines(True)

        version = int(Identity.parse_field("Version", lines[n]))
        n += 1

        Certification.parse_field("Type", lines[n])
        n += 1

        currency = Certification.parse_field("Currency", lines[n])
        n += 1

        pubkey_from = Certification.parse_field("Issuer", lines[n])
        n += 1

        identity_pubkey = Certification.parse_field("IdtyIssuer", lines[n])
        n += 1

        identity_uid = Certification.parse_field("IdtyUniqueID", lines[n])
        n += 1

        identity_timestamp = BlockUID.from_str(Certification.parse_field("IdtyTimestamp", lines[n]))
        n += 1

        identity_signature = Certification.parse_field("IdtySignature", lines[n])
        n += 1

        timestamp = BlockUID.from_str(Certification.parse_field("CertTimestamp", lines[n]))
        n += 1

        signature = Certification.parse_field("Signature", lines[n])

        identity = Identity(version, currency, identity_pubkey, identity_uid, identity_timestamp, identity_signature)

        return cls(version, currency, pubkey_from, identity, timestamp, signature)