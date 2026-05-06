def _parseRecord(self, gff3Set, line):
        """
        Parse one record.
        """
        row = line.split("\t")
        if len(row) != self.GFF3_NUM_COLS:
            raise GFF3Exception(
                "Wrong number of columns, expected {}, got {}".format(
                    self.GFF3_NUM_COLS, len(row)),
                self.fileName, self.lineNumber)
        feature = Feature(
            urllib.unquote(row[0]),
            urllib.unquote(row[1]),
            urllib.unquote(row[2]),
            int(row[3]), int(row[4]),
            row[5], row[6], row[7],
            self._parseAttrs(row[8]))
        gff3Set.add(feature)