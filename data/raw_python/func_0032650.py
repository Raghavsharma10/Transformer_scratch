def readQuotes(self, start, end):
        ''' read quotes '''
        rows = self.__hbase.scanTable(self.tableName(HBaseDAM.QUOTE), [HBaseDAM.QUOTE], start, end)

        return [self.__rowResultToQuote(row) for row in rows]