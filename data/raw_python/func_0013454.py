def readsGenerator(self, request):
        """
        Returns a generator over the (read, nextPageToken) pairs defined
        by the specified request
        """
        if not request.reference_id:
            raise exceptions.UnmappedReadsNotSupported()
        if len(request.read_group_ids) < 1:
            raise exceptions.BadRequestException(
                "At least one readGroupId must be specified")
        elif len(request.read_group_ids) == 1:
            return self._readsGeneratorSingle(request)
        else:
            return self._readsGeneratorMultiple(request)