def writeRow(self, row, values):
        '''
        write a row
        Not sure whether xlwt support write the same cell multiple times
        '''
        if self.__sheet is None:
            self.openSheet(super(ExcelWrite, self).DEFAULT_SHEET)

        for index, value in enumerate(values):
            self.__sheet.write(row, index, value)