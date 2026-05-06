def from_inline(cls: Type[BMAEndpointType], inline: str) -> BMAEndpointType:
        """
        Return BMAEndpoint instance from endpoint string

        :param inline: Endpoint string
        :return:
        """
        m = BMAEndpoint.re_inline.match(inline)
        if m is None:
            raise MalformedDocumentError(BMAEndpoint.API)
        server = m.group(1)
        ipv4 = m.group(2)
        ipv6 = m.group(3)
        port = int(m.group(4))
        return cls(server, ipv4, ipv6, port)