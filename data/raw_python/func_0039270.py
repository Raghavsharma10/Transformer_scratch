def metadata(self):
        """
        Write metadata sheet.
        """

        sheet = self.result.add_sheet("metadata")
        self.header(sheet, "metadata")

        n_row = 1  # row number

        for k in self.po.metadata:
            row = sheet.row(n_row)
            row.write(0, k)
            row.write(1, self.po.metadata[k])
            n_row += 1
            sheet.flush_row_data()