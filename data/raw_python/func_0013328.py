def parse(self):
        """
        Run the parse and return the resulting Gff3Set object.
        """
        fh = self._open()
        try:
            gff3Set = Gff3Set(self.fileName)
            for line in fh:
                self.lineNumber += 1
                self._parseLine(gff3Set, line[0:-1])
        finally:
            fh.close()
        gff3Set.linkChildFeaturesToParents()
        return gff3Set