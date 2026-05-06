def from_inline(cls: Type[UnknownEndpointType], inline: str) -> UnknownEndpointType:
        """
        Return UnknownEndpoint instance from endpoint string

        :param inline: Endpoint string
        :return:
        """
        try:
            api = inline.split()[0]
            properties = inline.split()[1:]
            return cls(api, properties)
        except IndexError:
            raise MalformedDocumentError(inline)