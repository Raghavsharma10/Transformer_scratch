def writeCell(self, row, col, value):
        ''' write a cell '''
        if self.__sheet is None:
            self.openSheet(super(ExcelWrite, self).DEFAULT_SHEET)

        self.__sheet.write(row, col, value)