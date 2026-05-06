def _parseNetDirectory(self, rva, size, magic = consts.PE32):
        """
        Parses the NET directory.
        @see: U{http://www.ntcore.com/files/dotnetformat.htm}
        
        @type rva: int 
        @param rva: The RVA where the NET directory starts.
        
        @type size: int
        @param size: The size of the NET directory.
        
        @type magic: int
        @param magic: (Optional) The type of PE. This value could be L{consts.PE32} or L{consts.PE64}.
        
        @rtype: L{NETDirectory}
        @return: A new L{NETDirectory} object.
        """        
        if not rva or not size:
            return None

        # create a NETDirectory class to hold the data
        netDirectoryClass = directories.NETDirectory()

        # parse the .NET Directory
        netDir = directories.NetDirectory.parse(utils.ReadData(self.getDataAtRva(rva,  size)))

        netDirectoryClass.directory = netDir

        # get the MetaData RVA and Size
        mdhRva = netDir.metaData.rva.value
        mdhSize = netDir.metaData.size.value

        # read all the MetaData
        rd = utils.ReadData(self.getDataAtRva(mdhRva, mdhSize))

        # parse the MetaData headers
        netDirectoryClass.netMetaDataHeader = directories.NetMetaDataHeader.parse(rd)

        # parse the NET metadata streams
        numberOfStreams = netDirectoryClass.netMetaDataHeader.numberOfStreams.value
        netDirectoryClass.netMetaDataStreams = directories.NetMetaDataStreams.parse(rd, numberOfStreams)

        for i in range(numberOfStreams):
            stream = netDirectoryClass.netMetaDataStreams[i]
            name = stream.name.value
            rd.setOffset(stream.offset.value)
            rd2 = utils.ReadData(rd.read(stream.size.value))
            stream.info = []
            if name == "#~" or i == 0:
                stream.info = rd2
            elif name == "#Strings" or i == 1:
                while len(rd2) > 0:
                    offset = rd2.tell()
                    stream.info.append({ offset: rd2.readDotNetString() })
            elif name == "#US" or i == 2:
                while len(rd2) > 0:
                    offset = rd2.tell()
                    stream.info.append({ offset: rd2.readDotNetUnicodeString() })
            elif name == "#GUID" or i == 3:
                while len(rd2) > 0:
                    offset = rd2.tell()
                    stream.info.append({ offset: rd2.readDotNetGuid() })
            elif name == "#Blob" or i == 4:
                while len(rd2) > 0:
                    offset = rd2.tell()
                    stream.info.append({ offset: rd2.readDotNetBlob() })

        for i in range(numberOfStreams):
            stream = netDirectoryClass.netMetaDataStreams[i]
            name = stream.name.value
            if name == "#~" or i == 0:
                stream.info = directories.NetMetaDataTables.parse(stream.info, netDirectoryClass.netMetaDataStreams)

        # parse .NET resources
        # get the Resources RVA and Size
        resRva = netDir.resources.rva.value
        resSize = netDir.resources.size.value

        # read all the MetaData
        rd = utils.ReadData(self.getDataAtRva(resRva, resSize))

        resources = []

        for i in netDirectoryClass.netMetaDataStreams[0].info.tables["ManifestResource"]:
            offset = i["offset"]
            rd.setOffset(offset)
            size = rd.readDword()
            data = rd.read(size)
            if data[:4] == "\xce\xca\xef\xbe":
                data = directories.NetResources.parse(utils.ReadData(data))
            resources.append({ "name": i["name"], "offset": offset + 4, "size": size, "data": data })

        netDirectoryClass.directory.resources.info = resources

        return netDirectoryClass