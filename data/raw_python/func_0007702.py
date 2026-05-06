def remove_row(self, row_number: int=-1):
        """
        Removes a specified row of data

        :param row_number: the row to remove (defaults to the last row)
        :return: None
        """
        if len(self._rows) == 0:
            return

        row = self._rows.pop(row_number)
        for widget in row:
            widget.destroy()