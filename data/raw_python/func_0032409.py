def readQuotes(self, start, end):
        ''' read quotes '''
        quotes = self.__readData(self.targetPath(ExcelDAM.QUOTE), start, end)
        return [Quote(*quote) for quote in quotes]