def runGetRequest(self, obj):
        """
        Runs a get request by converting the specified datamodel
        object into its protocol representation.
        """
        protocolElement = obj.toProtocolElement()
        jsonString = protocol.toJson(protocolElement)
        return jsonString