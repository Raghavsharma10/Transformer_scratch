def from_inline(cls: Type[ESUserEndpointType], inline: str) -> ESUserEndpointType:
        """
        Return ESUserEndpoint instance from endpoint string

        :param inline: Endpoint string
        :return:
        """
        m = ESUserEndpoint.re_inline.match(inline)
        if m is None:
            raise MalformedDocumentError(ESUserEndpoint.API)
        server = m.group(1)
        port = int(m.group(2))
        return cls(server, port)