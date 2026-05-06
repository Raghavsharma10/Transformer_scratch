def _init(self):
        """
        initialize all values based on provided input

        :return: None
        """
        self.col_count = len(self.col_list)
        # list of lengths of longest entries in columns
        self.col_longest = self.get_all_longest_col_lengths()
        self.data_length = sum(self.col_longest.values())

        if self.terminal_width > 0:
            # free space is space which should be equeally distributed for all columns
            # self.terminal_width -- terminal is our canvas
            #  - self.data_length -- substract length of content (the actual data)
            #  - self.col_count + 1 -- table lines are not part of free space, their width is
            #                          (number of columns - 1)
            self.total_free_space = (self.terminal_width - self.data_length) - self.col_count + 1
            if self.total_free_space <= 0:
                self.total_free_space = None
            else:
                self.default_column_space = self.total_free_space // self.col_count
                self.default_column_space_remainder = self.total_free_space % self.col_count
                logger.debug("total free space: %d, column space: %d, remainder: %d, columns: %d",
                             self.total_free_space, self.default_column_space,
                             self.default_column_space_remainder, self.col_count)
        else:
            self.total_free_space = None