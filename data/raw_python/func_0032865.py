def constructRows(self, items):
        """
        Build row objects that are serializable using Athena for sending to the
        client.

        @param items: an iterable of objects compatible with my columns'
        C{extractValue} methods.

        @return: a list of dictionaries, where each dictionary has a string key
        for each column name in my list of columns.
        """
        rows = []
        for item in items:
            row = dict((colname, col.extractValue(self, item))
                       for (colname, col) in self.columns.iteritems())
            link = self.linkToItem(item)
            if link is not None:
                row[u'__id__'] = link
            rows.append(row)

        return rows