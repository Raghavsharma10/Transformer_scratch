def write_quotes(self, quotes):
        ''' write quotes '''
        if self.first:
            Base.metadata.create_all(self.engine, checkfirst=True)
            self.first=False

        session=self.getWriteSession()
        session.add_all([self.__quoteToSql(quote) for quote in quotes])