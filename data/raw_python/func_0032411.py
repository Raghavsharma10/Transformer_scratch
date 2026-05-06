def readTicks(self, start, end):
        ''' read ticks '''
        ticks =  self.__readData(self.targetPath(ExcelDAM.TICK), start, end)
        return [Tick(*tick) for tick in ticks]