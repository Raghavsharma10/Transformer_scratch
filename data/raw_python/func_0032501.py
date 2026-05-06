def _extractCreations(self, dataSets):
        """
        Find the elements of C{dataSets} which represent the creation of new
        objects.

        @param dataSets: C{list} of C{dict} mapping C{unicode} form submission
            keys to form submission values.

        @return: iterator of C{tuple}s with the first element giving the opaque
            identifier of an object which is to be created and the second
            element giving a C{dict} of all the other creation arguments.
        """
        for dataSet in dataSets:
            modelObject = self._objectFromID(dataSet[self._IDENTIFIER_KEY])
            if modelObject is self._NO_OBJECT_MARKER:
                dataCopy = dataSet.copy()
                identifier = dataCopy.pop(self._IDENTIFIER_KEY)
                yield identifier, dataCopy