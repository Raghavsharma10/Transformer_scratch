def get_all_longest_col_lengths(self):
        """
        iterate over all columns and get their longest values

        :return: dict, {"column_name": 132}
        """
        response = {}
        for col in self.col_list:
            response[col] = self._longest_val_in_column(col)
        return response