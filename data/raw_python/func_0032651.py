def writeQuotes(self, quotes):
        ''' write quotes '''
        tName = self.tableName(HBaseDAM.QUOTE)
        if tName not in self.__hbase.getTableNames():
            self.__hbase.createTable(tName, [ColumnDescriptor(name=HBaseDAM.QUOTE, maxVersions=5)])

        for quote in quotes:
            self.__hbase.updateRow(self.tableName(HBaseDAM.QUOTE),
                                   quote.time,
                                   [Mutation(column = "%s:%s" % (HBaseDAM.QUOTE, field),
                                             value = getattr(quote, field) ) for field in QUOTE_FIELDS])