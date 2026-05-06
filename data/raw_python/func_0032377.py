def __getSheet(self, name):
        ''' get a sheet by name '''
        if not self.sheetExsit(name):
            raise UfException(Errors.SHEET_NAME_INVALID, "Can't find a sheet named %s" % name)

        return self.__sheetNameDict[name]