def parse(readDataInstance):
        """
        Returns a new L{NetResources} object.

        @type readDataInstance: L{ReadData}
        @param readDataInstance: A L{ReadData} object with data to be parsed as a L{NetResources} object.

        @rtype: L{NetResources}
        @return: A new L{NetResources} object.
        """
        r = NetResources()

        r.signature = readDataInstance.readDword()
        if r.signature != 0xbeefcace:
            return r

        r.readerCount = readDataInstance.readDword()
        r.readerTypeLength = readDataInstance.readDword()
        r.readerType = utils.ReadData(readDataInstance.read(r.readerTypeLength)).readDotNetBlob()
        r.version = readDataInstance.readDword()
        r.resourceCount = readDataInstance.readDword()
        r.resourceTypeCount = readDataInstance.readDword()

        r.resourceTypes = []
        for i in xrange(r.resourceTypeCount):
            r.resourceTypes.append(readDataInstance.readDotNetBlob())

        # aligned to 8 bytes
        readDataInstance.skipBytes(8 - readDataInstance.tell() & 0x7)

        r.resourceHashes = []
        for i in xrange(r.resourceCount):
            r.resourceHashes.append(readDataInstance.readDword())

        r.resourceNameOffsets = []
        for i in xrange(r.resourceCount):
            r.resourceNameOffsets.append(readDataInstance.readDword())

        r.dataSectionOffset = readDataInstance.readDword()

        r.resourceNames = []
        r.resourceOffsets = []
        base = readDataInstance.tell()
        for i in xrange(r.resourceCount):
            readDataInstance.setOffset(base + r.resourceNameOffsets[i])
            r.resourceNames.append(readDataInstance.readDotNetUnicodeString())
            r.resourceOffsets.append(readDataInstance.readDword())

        r.info = {}
        for i in xrange(r.resourceCount):
            readDataInstance.setOffset(r.dataSectionOffset + r.resourceOffsets[i])
            r.info[i] = readDataInstance.read(len(readDataInstance))
            r.info[r.resourceNames[i]] = r.info[i]

        return r