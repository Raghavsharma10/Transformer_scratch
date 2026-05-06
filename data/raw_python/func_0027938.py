def prepareInsert(self, oself, store):
        """
        Prepare for insertion into the database by making the dbunderlying
        attribute of the item a relative pathname with respect to the store
        rather than an absolute pathname.
        """
        if self.relative:
            fspath = self.__get__(oself)
            oself.__dirty__[self.attrname] = self, self.infilter(fspath, oself, store)