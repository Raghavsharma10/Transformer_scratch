def writeTicks(self, ticks):
        ''' read quotes '''
        tName = self.tableName(HBaseDAM.TICK)
        if tName not in self.__hbase.getTableNames():
            self.__hbase.createTable(tName, [ColumnDescriptor(name=HBaseDAM.TICK, maxVersions=5)])

        for tick in ticks:
            self.__hbase.updateRow(self.tableName(HBaseDAM.TICK),
                                   tick.time,
                                   [Mutation(column = "%s:%s" % (HBaseDAM.TICK, field),
                                             value = getattr(tick, field) ) for field in TICK_FIELDS])