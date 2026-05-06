def runGetInfo(self, request):
        """
        Returns information about the service including protocol version.
        """
        return protocol.toJson(protocol.GetInfoResponse(
            protocol_version=protocol.version))