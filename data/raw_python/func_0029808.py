def row(self, fields):
        """Return a row for fields, for CSV files, pretty printing, etc, give a set of fields to return"""

        d = self.dict

        row = [None] * len(fields)

        for i, f in enumerate(fields):
            if f in d:
                row[i] = d[f]

        return row