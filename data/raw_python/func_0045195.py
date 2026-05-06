def _transform_rows(self):
        """
        Transforms all source rows.
        """
        self._find_all_step_methods()

        for row in self._source_reader.next():
            self._transform_row_wrapper(row)