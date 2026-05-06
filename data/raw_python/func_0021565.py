def parse(readDataInstance, netMetaDataStreams):
        """
        Returns a new L{NetMetaDataTables} object.
        
        @type readDataInstance: L{ReadData}
        @param readDataInstance: A L{ReadData} object with data to be parsed as a L{NetMetaDataTables} object.
        
        @rtype: L{NetMetaDataTables}
        @return: A new L{NetMetaDataTables} object.
        """
        dt = NetMetaDataTables()
        dt.netMetaDataTableHeader = NetMetaDataTableHeader.parse(readDataInstance)
        dt.tables = {}

        metadataTableDefinitions = dotnet.MetadataTableDefinitions(dt, netMetaDataStreams)

        for i in xrange(64):
            dt.tables[i] = { "rows": 0 }
            if dt.netMetaDataTableHeader.maskValid.value >> i & 1:
                dt.tables[i]["rows"] = readDataInstance.readDword()
            if i in dotnet.MetadataTableNames:
                dt.tables[dotnet.MetadataTableNames[i]] = dt.tables[i]

        for i in xrange(64):
            dt.tables[i]["data"] = []
            for j in range(dt.tables[i]["rows"]):
                row = None
                if i in metadataTableDefinitions:
                    row = readDataInstance.readFields(metadataTableDefinitions[i])
                dt.tables[i]["data"].append(row)

        for i in xrange(64):
            if i in dotnet.MetadataTableNames:
                dt.tables[dotnet.MetadataTableNames[i]] = dt.tables[i]["data"]
            dt.tables[i] = dt.tables[i]["data"]

        return dt