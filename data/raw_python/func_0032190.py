def write_ticks(self, ticks):
        ''' write ticks '''
        if self.first:
            Base.metadata.create_all(self.engine, checkfirst=True)
            self.first=False

        session=self.getWriteSession()
        session.add_all([self.__tickToSql(tick) for tick in ticks])