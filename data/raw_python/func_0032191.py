def write_fundamental(self, keyTimeValueDict):
        ''' write fundamental '''
        if self.first:
            Base.metadata.create_all(self.__getEngine(), checkfirst=True)
            self.first=False

        sqls=self._fundamentalToSqls(keyTimeValueDict)
        session=self.Session()
        try:
            session.add_all(sqls)
        finally:
            self.Session.remove()