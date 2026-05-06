def push(self,bookName=None,sheetName=None,overwrite=False):
        """pull this OR.SHEET into a real book/sheet in Origin"""
        # tons of validation
        if bookName: self.bookName=bookName
        if sheetName: self.sheetName=sheetName
        if not self.sheetName in OR.sheetNames(bookName):
            print("can't find [%s]%s!"%(bookName,sheetName))
            return

        # clear out out sheet by deleting EVERY column
        poSheet=OR.getSheet(bookName,sheetName) # CPyWorksheetPageI
        if not poSheet:
            print("WARNING: didn't get posheet",poSheet,bookName,sheetName)
        for poCol in [x for x in poSheet if x.IsValid()]:
            poCol.Destroy()

        # create columns and assign properties to each
        for i in range(len(self.colNames)):
            poSheet.InsertCol(i,self.colNames[i])
            poSheet.Columns(i).SetName(self.colNames[i])
            poSheet.Columns(i).SetLongName(self.colDesc[i])
            poSheet.Columns(i).SetUnits(self.colUnits[i])
            poSheet.Columns(i).SetComments(self.colComments[i])
            poSheet.Columns(i).SetType(self.colTypes[i])
            poSheet.Columns(i).SetData(self.colData[i])