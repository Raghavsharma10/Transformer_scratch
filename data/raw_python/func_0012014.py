def makeProducer(self, request, fileForReading):
        """
        Make a L{StaticProducer} that will produce the body of this response.

        This method will also set the response code and Content-* headers.

        @param request: The L{Request} object.
        @param fileForReading: The file object containing the resource.
        @return: A L{StaticProducer}.  Calling C{.start()} on this will begin
            producing the response.
        """
        byteRange = request.getHeader(b'range')
        if byteRange is None or not self.getFileSize():
            self._setContentHeaders(request)
            request.setResponseCode(http.OK)
            return NoRangeStaticProducer(request, fileForReading)
        try:
            parsedRanges = self._parseRangeHeader(byteRange)
        except ValueError:
            logger.warning("Ignoring malformed Range header %r" % (byteRange,))
            self._setContentHeaders(request)
            request.setResponseCode(http.OK)
            return NoRangeStaticProducer(request, fileForReading)

        if len(parsedRanges) == 1:
            offset, size = self._doSingleRangeRequest(
                request, parsedRanges[0])
            self._setContentHeaders(request, size)
            return SingleRangeStaticProducer(
                request, fileForReading, offset, size)
        else:
            rangeInfo = self._doMultipleRangeRequest(request, parsedRanges)
            return MultipleRangeStaticProducer(
                request, fileForReading, rangeInfo)