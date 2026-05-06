def getColumn(self, attributeName, raw=False):
        """
        Get an L{iaxiom.IQuery} whose results will be values of a single
        attribute rather than an Item.

        @param attributeName: a L{str}, the name of a Python attribute, that
        describes a column on the Item subclass that this query was specified
        for.

        @return: an L{AttributeQuery} for the column described by the attribute
        named L{attributeName} on the item class that this query's results will
        be instances of.
        """
        # XXX: 'raw' is undocumented because I think it's completely unused,
        # and it's definitely untested.  It should probably be removed when
        # someone has the time. -glyph

        # Quotient POP3 server uses it.  Not that it shouldn't be removed.
        # ;) -exarkun
        attr = getattr(self.tableClass, attributeName)
        return AttributeQuery(self.store,
                              self.tableClass,
                              self.comparison,
                              self.limit,
                              self.offset,
                              self.sort,
                              attr,
                              raw)