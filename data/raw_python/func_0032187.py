def readBatchTupleQuotes(self, symbols, start, end):
        '''
        read batch quotes as tuple to save memory
        '''
        if end is None:
            end=sys.maxint

        ret={}
        session=self.getReadSession()()
        try:
            symbolChunks=splitListEqually(symbols, 100)
            for chunk in symbolChunks:
                rows=session.query(Quote.symbol, Quote.time, Quote.close, Quote.volume,
                                     Quote.low, Quote.high).filter(and_(Quote.symbol.in_(chunk),
                                                                              Quote.time >= int(start),
                                                                              Quote.time < int(end)))

                for row in rows:
                    if row.time not in ret:
                        ret[row.time]={}

                    ret[row.time][row.symbol]=self.__sqlToTupleQuote(row)
        finally:
            self.getReadSession().remove()

        return ret