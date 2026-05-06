def from_signed_raw(cls: Type[PeerType], raw: str) -> PeerType:
        """
        Return a Peer instance from a signed raw format string

        :param raw: Signed raw format string
        :return:
        """
        lines = raw.splitlines(True)
        n = 0

        version = int(Peer.parse_field("Version", lines[n]))
        n += 1

        Peer.parse_field("Type", lines[n])
        n += 1

        currency = Peer.parse_field("Currency", lines[n])
        n += 1

        pubkey = Peer.parse_field("Pubkey", lines[n])
        n += 1

        block_uid = BlockUID.from_str(Peer.parse_field("Block", lines[n]))
        n += 1

        Peer.parse_field("Endpoints", lines[n])
        n += 1

        endpoints = []
        while not Peer.re_signature.match(lines[n]):
            endpoints.append(endpoint(lines[n]))
            n += 1

        data = Peer.re_signature.match(lines[n])
        if data is None:
            raise MalformedDocumentError("Peer")
        signature = data.group(1)

        return cls(version, currency, pubkey, block_uid, endpoints, signature)