def from_inline(cls: Type[WS2PEndpointType], inline: str) -> WS2PEndpointType:
        """
        Return WS2PEndpoint instance from endpoint string

        :param inline: Endpoint string
        :return:
        """
        m = WS2PEndpoint.re_inline.match(inline)
        if m is None:
            raise MalformedDocumentError(WS2PEndpoint.API)
        ws2pid = m.group(1)
        server = m.group(2)
        port = int(m.group(3))
        path = m.group(4)
        if not path:
            path = ""
        return cls(ws2pid, server, port, path)