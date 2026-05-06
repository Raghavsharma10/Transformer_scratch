def runListReferenceBases(self, requestJson):
        """
        Runs a listReferenceBases request for the specified ID and
        request arguments.
        """
        # In the case when an empty post request is made to the endpoint
        # we instantiate an empty ListReferenceBasesRequest.
        if not requestJson:
            request = protocol.ListReferenceBasesRequest()
        else:
            try:
                request = protocol.fromJson(
                    requestJson,
                    protocol.ListReferenceBasesRequest)
            except protocol.json_format.ParseError:
                raise exceptions.InvalidJsonException(requestJson)
        compoundId = datamodel.ReferenceCompoundId.parse(request.reference_id)
        referenceSet = self.getDataRepository().getReferenceSet(
            compoundId.reference_set_id)
        reference = referenceSet.getReference(request.reference_id)
        start = request.start
        end = request.end
        if end == 0:  # assume meant "get all"
            end = reference.getLength()
        if request.page_token:
            pageTokenStr = request.page_token
            start = paging._parsePageToken(pageTokenStr, 1)[0]

        chunkSize = self._maxResponseLength
        nextPageToken = None
        if start + chunkSize < end:
            end = start + chunkSize
            nextPageToken = str(start + chunkSize)
        sequence = reference.getBases(start, end)

        # build response
        response = protocol.ListReferenceBasesResponse()
        response.offset = start
        response.sequence = sequence
        if nextPageToken:
            response.next_page_token = nextPageToken
        return protocol.toJson(response)