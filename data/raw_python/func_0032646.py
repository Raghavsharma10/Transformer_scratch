def _decompose(self, value):
        """
        Decompose an instance of our record type into a dictionary mapping
        attribute names to values.

        @param value: an instance of self.recordType

        @return: L{dict} containing the keys declared on L{record}.
        """
        data = {}
        for n, attr in zip(self.recordType.__names__, self.attrs):
            data[attr.attrname] = getattr(value, n)
        return data