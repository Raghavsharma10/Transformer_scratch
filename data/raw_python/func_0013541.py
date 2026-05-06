def populateFromRow(self, readGroupSetRecord):
        """
        Populates the instance variables of this ReadGroupSet from the
        specified database row.
        """
        self._dataUrl = readGroupSetRecord.dataurl
        self._indexFile = readGroupSetRecord.indexfile
        self._programs = []
        for jsonDict in json.loads(readGroupSetRecord.programs):
            program = protocol.fromJson(json.dumps(jsonDict),
                                        protocol.Program)
            self._programs.append(program)
        stats = protocol.fromJson(readGroupSetRecord.stats, protocol.ReadStats)
        self._numAlignedReads = stats.aligned_read_count
        self._numUnalignedReads = stats.unaligned_read_count