def readCell(self, row, col):
        ''' read a cell'''
        try:
            if self.__sheet is None:
                self.openSheet(super(ExcelRead, self).DEFAULT_SHEET)

            return self.__sheet.cell(row, col).value
        except BaseException as excp:
            raise UfException(Errors.UNKNOWN_ERROR, "Unknown Error in Excellib.readCell %s" % excp)