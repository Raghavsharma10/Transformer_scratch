def getSerializedResponse(self):
        """
        Returns a string version of the SearchResponse that has
        been built by this SearchResponseBuilder.
        """
        self._protoObject.next_page_token = pb.string(self._nextPageToken)
        s = protocol.toJson(self._protoObject)
        return s