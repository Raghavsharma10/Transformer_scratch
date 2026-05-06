def _separate(self):
        """
        get a width of separator for current column

        :return: int
        """
        if self.total_free_space is None:
            return 0
        else:
            sepa = self.default_column_space
            # we need to distribute remainders
            if self.default_column_space_remainder > 0:
                sepa += 1
                self.default_column_space_remainder -= 1
            logger.debug("remainder: %d, separator: %d",
                         self.default_column_space_remainder, sepa)
            return sepa