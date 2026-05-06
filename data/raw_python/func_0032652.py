def readTicks(self, start, end):
        ''' read ticks '''
        rows = self.__hbase.scanTable(self.tableName(HBaseDAM.TICK), [HBaseDAM.TICK], start, end)
        return [self.__rowResultToTick(row) for row in rows]