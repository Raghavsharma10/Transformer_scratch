def extract_self_cert(signed_raw: str) -> Identity:
        """
        Return self-certified Identity instance from the signed raw Revocation document

        :param signed_raw: Signed raw document string
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

        unique_id = Revocation.parse_field("IdtyUniqueID", lines[n])
        n += 1

        timestamp = Revocation.parse_field("IdtyTimestamp", lines[n])
        n += 1

        signature = Revocation.parse_field("IdtySignature", lines[n])
        n += 1

        return Identity(version, currency, issuer, unique_id, timestamp, signature)