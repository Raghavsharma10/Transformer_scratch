def from_inline(cls: Type[SecuredBMAEndpointType], inline: str) -> SecuredBMAEndpointType:
        """
        Return SecuredBMAEndpoint instance from endpoint string

        :param inline: Endpoint string
        :return:
        """
        m = SecuredBMAEndpoint.re_inline.match(inline)
        if m is None:
            raise MalformedDocumentError(SecuredBMAEndpoint.API)
        server = m.group(1)
        ipv4 = m.group(2)
        ipv6 = m.group(3)
        port = int(m.group(4))
        path = m.group(5)
        if not path:
            path = ""
        return cls(server, ipv4, ipv6, port, path)