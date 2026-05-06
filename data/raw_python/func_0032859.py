def rowsBeforeRow(self, rowObject, count):
        """
        Wrapper around L{rowsBeforeItem} which accepts the web ID for a item
        instead of the item itself.

        @param rowObject: a dictionary mapping strings to column values, sent
        from the client.  One of those column values must be C{__id__} to
        uniquely identify a row.

        @param count: an integer, the number of rows to return.
        """
        webID = rowObject['__id__']
        return self.rowsBeforeItem(
            self.webTranslator.fromWebID(webID),
            count)