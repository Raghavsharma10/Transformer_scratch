def writeQuotes(self, quotes):
        ''' write quotes '''
        self.__writeData(self.targetPath(ExcelDAM.QUOTE),
                         QUOTE_FIELDS,
                         [[getattr(quote, field) for field in QUOTE_FIELDS] for quote in quotes])