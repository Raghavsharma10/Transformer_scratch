def new_sheet(self, name=None, cols=None):
        """
        Create a new sheet in the spreadsheet and return it so content can be added.
        :param name: Optional name for the sheet.
        :param cols: Specify the number of columns, needed for compatibility in some cases
        :return: Sheet object
        """
        sheet = Sheet(self.dom, name, cols)
        self.sheets.append(sheet)
        return sheet