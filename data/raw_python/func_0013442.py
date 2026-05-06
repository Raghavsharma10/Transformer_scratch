def _topLevelObjectGenerator(self, request, numObjects, getByIndexMethod):
        """
        Returns a generator over the results for the specified request, which
        is over a set of objects of the specified size. The objects are
        returned by call to the specified method, which must take a single
        integer as an argument. The returned generator yields a sequence of
        (object, nextPageToken) pairs, which allows this iteration to be picked
        up at any point.
        """
        currentIndex = 0
        if request.page_token:
            currentIndex, = paging._parsePageToken(
                request.page_token, 1)
        while currentIndex < numObjects:
            object_ = getByIndexMethod(currentIndex)
            currentIndex += 1
            nextPageToken = None
            if currentIndex < numObjects:
                nextPageToken = str(currentIndex)
            yield object_.toProtocolElement(), nextPageToken