def openSheet(self, name):
        ''' set a sheet to write '''
        if name not in self.__sheetNameDict:
            sheet = self.__workbook.add_sheet(name)
            self.__sheetNameDict[name] = sheet

        self.__sheet = self.__sheetNameDict[name]