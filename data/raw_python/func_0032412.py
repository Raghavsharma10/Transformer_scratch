def writeTicks(self, ticks):
        ''' read quotes '''
        self.__writeData(self.targetPath(ExcelDAM.TICK),
                         TICK_FIELDS,
                         [[getattr(tick, field) for field in TICK_FIELDS] for tick in ticks])