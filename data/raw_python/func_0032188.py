def read_tuple_ticks(self, symbol, start, end):
        ''' read ticks as tuple '''
        if end is None:
            end=sys.maxint

        session=self.getReadSession()()
        try:
            rows=session.query(Tick).filter(and_(Tick.symbol == symbol,
                                                      Tick.time >= int(start),
                                                      Tick.time < int(end)))
        finally:
            self.getReadSession().remove()

        return [self.__sqlToTupleTick(row) for row in rows]