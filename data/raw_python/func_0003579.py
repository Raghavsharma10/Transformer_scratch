def row_completed(self, index):
        """Mark the row at index as completed.

        .. seealso:: :meth:`completed_row_indices`

        This method notifies the obsevrers from :meth:`on_row_completed`.
        """
        self._completed_rows.append(index)
        for row_completed in self._on_row_completed:
            row_completed(index)