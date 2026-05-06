def readTupleQuotes(self, symbol, start, end):
        ''' read quotes as tuple '''
        if end is None:
            end=sys.maxint

        session=self.getReadSession()()
        try:
            rows=session.query(Quote).filter(and_(Quote.symbol == symbol,
                                                       Quote.time >= int(start),
                                                       Quote.time < int(end)))
        finally:
            self.getReadSession().remove()

        return rows