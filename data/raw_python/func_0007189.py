def _open_sheet(self):
        """
        Read the sheet, get value the header, get number columns and rows
        :return:
        """
        if self.sheet_name and not self.header:
            self._sheet = self._file.worksheet(self.sheet_name.title)
            self.ncols = self._sheet.col_count
            self.nrows = self._sheet.row_count
            for i in range(1, self.ncols+1):
                self.header = self.header + [self._sheet.cell(1, i).value]