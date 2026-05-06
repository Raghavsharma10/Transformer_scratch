def from_inline(cls: Type[ESCoreEndpointType], inline: str) -> ESCoreEndpointType:
        """
        Return ESCoreEndpoint instance from endpoint string

        :param inline: Endpoint string
        :return:
        """
        m = ESCoreEndpoint.re_inline.match(inline)
        if m is None:
            raise MalformedDocumentError(ESCoreEndpoint.API)
        server = m.group(1)
        port = int(m.group(2))
        return cls(server, port)