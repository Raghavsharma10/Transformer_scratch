def _readGroupSetsGenerator(self, request, numObjects, getByIndexMethod):
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
            obj = getByIndexMethod(currentIndex)
            include = True
            rgsp = obj.toProtocolElement()
            if request.name and request.name != obj.getLocalId():
                include = False
            if request.biosample_id and include:
                rgsp.ClearField("read_groups")
                for readGroup in obj.getReadGroups():
                    if request.biosample_id == readGroup.getBiosampleId():
                        rgsp.read_groups.extend(
                            [readGroup.toProtocolElement()])
                # If none of the biosamples match and the readgroupset
                # contains reagroups, don't include in the response
                if len(rgsp.read_groups) == 0 and \
                        len(obj.getReadGroups()) != 0:
                    include = False
            currentIndex += 1
            nextPageToken = None
            if currentIndex < numObjects:
                nextPageToken = str(currentIndex)
            if include:
                yield rgsp, nextPageToken