def addValue(self, protocolElement):
        """
        Appends the specified protocolElement to the value list for this
        response.
        """
        self._numElements += 1
        self._bufferSize += protocolElement.ByteSize()
        attr = getattr(self._protoObject, self._valueListName)
        obj = attr.add()
        obj.CopyFrom(protocolElement)