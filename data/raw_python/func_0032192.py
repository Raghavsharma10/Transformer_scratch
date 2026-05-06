def _fundamentalToSqls(self, symbol, keyTimeValueDict):
        ''' convert fundament dict to sqls '''
        sqls=[]
        for key, timeValues in keyTimeValueDict.iteritems():
            for timeStamp, value in timeValues.iteritems():
                sqls.append(FmSql(symbol, key, timeStamp, value))

        return sqls