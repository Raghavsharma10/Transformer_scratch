def _step99(self, in_row, tmp_row, out_row):
        """
        Validates all mandatory fields are in the output row and are filled.

        :param dict in_row: The input row.
        :param dict tmp_row: Not used.
        :param dict out_row: The output row.
        """
        park_info = ''
        for field in self._mandatory_fields:
            if field not in out_row or not out_row[field]:
                if park_info:
                    park_info += ' '
                park_info += field

        return park_info, None