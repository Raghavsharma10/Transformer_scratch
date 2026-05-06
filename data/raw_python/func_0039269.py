def strings(self):
        """
        Write strings sheet.
        """

        sheet = self.result.add_sheet("strings")
        self.header(sheet, "strings")

        n_row = 1  # row number

        for entry in self.po:
            row = sheet.row(n_row)
            row.write(0, entry.msgid)
            row.write(1, entry.msgstr)
            n_row += 1
            sheet.flush_row_data()