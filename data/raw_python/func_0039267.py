def header(self, sheet, name):
        """
        Write sheet header.
        Args:
            sheet: (xlwt.Worksheet.Worksheet) instance of xlwt sheet.
            name: (unicode) name of sheet.
        """

        header = sheet.row(0)
        for i, column in enumerate(self.headers[name]):
            header.write(i, self.headers[name][i])