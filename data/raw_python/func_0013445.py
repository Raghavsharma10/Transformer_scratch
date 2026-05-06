def _objectListGenerator(self, request, objectList):
        """
        Returns a generator over the objects in the specified list using
        _topLevelObjectGenerator to generate page tokens.
        """
        return self._topLevelObjectGenerator(
            request, len(objectList), lambda index: objectList[index])