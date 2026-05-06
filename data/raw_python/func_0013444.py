def _protocolListGenerator(self, request, objectList):
        """
        Returns a generator over the objects in the specified list using
        _protocolObjectGenerator to generate page tokens.
        """
        return self._protocolObjectGenerator(
            request, len(objectList), lambda index: objectList[index])