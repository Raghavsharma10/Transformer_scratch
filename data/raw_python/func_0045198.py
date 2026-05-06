def _step00(self, in_row, tmp_row, out_row):
        """
        Prunes whitespace for all fields in the input row.

        :param dict in_row: The input row.
        :param dict tmp_row: Not used.
        :param dict out_row: Not used.
        """
        for key, value in in_row.items():
            in_row[key] = WhitespaceCleaner.clean(value)

        return None, None