def readQuotes(self, start, end):
        ''' read quotes from Yahoo Financial'''
        if self.symbol is None:
            LOG.debug('Symbol is None')
            return []

        return self.__yf.getQuotes(self.symbol, start, end)