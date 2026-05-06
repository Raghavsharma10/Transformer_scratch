def _massageData(self, row):
        """
        Convert a raw database row to the type described by an attribute.  For
        example, convert a database integer into an L{extime.Time} instance for
        an L{attributes.timestamp} attribute.

        @param row: a 1-tuple, containing the in-database value from my
        attribute.

        @return: a value of the type described by my attribute.
        """
        if self.raw:
            return row[0]
        return self.attribute.outfilter(row[0], _FakeItemForFilter(self.store))