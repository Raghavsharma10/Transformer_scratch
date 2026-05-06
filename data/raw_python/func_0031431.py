def saveAsJSON(self, fp):
        """
        Write the records out as JSON. The first JSON object saved contains
        the BLAST parameters.

        @param fp: A C{str} file pointer to write to.
        """
        first = True
        for record in self.records():
            if first:
                print(dumps(self.params, separators=(',', ':')), file=fp)
                first = False
            print(dumps(self._convertBlastRecordToDict(record),
                        separators=(',', ':')), file=fp)