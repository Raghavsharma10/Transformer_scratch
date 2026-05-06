def getVariants(self, referenceName, startPosition, endPosition,
                    callSetIds=[]):
        """
        Returns an iterator over the specified variants. The parameters
        correspond to the attributes of a GASearchVariantsRequest object.
        """
        if callSetIds is None:
            callSetIds = self._callSetIds
        else:
            for callSetId in callSetIds:
                if callSetId not in self._callSetIds:
                    raise exceptions.CallSetNotInVariantSetException(
                        callSetId, self.getId())
        for record in self.getPysamVariants(
                referenceName, startPosition, endPosition):
            yield self.convertVariant(record, callSetIds)