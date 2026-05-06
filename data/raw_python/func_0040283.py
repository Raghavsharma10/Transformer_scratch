def from_inline(cls: Type[ESSubscribtionEndpointType], inline: str) -> ESSubscribtionEndpointType:
        """
        Return ESSubscribtionEndpoint instance from endpoint string

        :param inline: Endpoint string
        :return:
        """
        m = ESSubscribtionEndpoint.re_inline.match(inline)
        if m is None:
            raise MalformedDocumentError(ESSubscribtionEndpoint.API)
        server = m.group(1)
        port = int(m.group(2))
        return cls(server, port)