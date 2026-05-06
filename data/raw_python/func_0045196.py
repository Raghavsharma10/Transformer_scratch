def _transform_row_wrapper(self, row):
        """
        Transforms a single source row.

        :param dict[str|str] row: The source row.
        """
        self._count_total += 1

        try:
            # Transform the naturals keys in line to technical keys.
            in_row = copy.copy(row)
            out_row = {}
            park_info, ignore_info = self._transform_row(in_row, out_row)

        except Exception as e:
            # Log the exception.
            self._handle_exception(row, e)
            # Keep track of the number of errors.
            self._count_error += 1
            # This row must be parked.
            park_info = 'Exception'
            # Keep our IDE happy.
            ignore_info = None
            out_row = {}

        if park_info:
            # Park the row.
            self.pre_park_row(park_info, row)
            self._parked_writer.writerow(row)
            self._count_park += 1
        elif ignore_info:
            # Ignore the row.
            self.pre_ignore_row(ignore_info, row)
            self._ignored_writer.writerow(row)
            self._count_ignore += 1
        else:
            # Write the technical keys and measures to the output file.
            self._transformed_writer.writerow(out_row)
            self._count_transform += 1