def _longest_val_in_column(self, col):
        """
        get size of longest value in specific column

        :param col: str, column name
        :return int
        """
        try:
            # +2 is for implicit separator
            return max([len(x[col]) for x in self.table if x[col]]) + 2
        except KeyError:
            logger.error("there is no column %r", col)
            raise