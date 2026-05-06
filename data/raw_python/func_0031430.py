def records(self):
        """
        Yield BLAST records, as read by the BioPython NCBIXML.parse
        method. Set self.params from data in the first record.
        """
        first = True
        with as_handle(self._filename) as fp:
            for record in NCBIXML.parse(fp):
                if first:
                    self.params = self._convertBlastParamsToDict(record)
                    first = False
                yield record