def toProtocolElement(self):
        """
        Converts this exception into the GA4GH protocol type so that
        it can be communicated back to the client.
        """
        error = protocol.GAException()
        error.error_code = self.getErrorCode()
        error.message = self.getMessage()
        return error