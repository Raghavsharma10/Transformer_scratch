def _count_sizes(self):
        """
        count all values needed to display whole table

        <><---terminal-width-----------><>

        <> HEADER  | HEADER2  | HEADER3 <>
        <>---------+----------+---------<>

        kudos to PostgreSQL developers

        :return: None
        """
        format_list = []
        header_sepa_format_list = []
        # actual widths of columns
        self.col_widths = {}

        for col in self.col_list:
            col_length = self.col_longest[col]
            col_width = col_length + self._separate()
            # -2 is for implicit separator -- spaces around
            format_list.append(" {%s:%d} " % (col, col_width - 2))
            header_sepa_format_list.append("{%s:%d}" % (col, col_width))
            self.col_widths[col] = col_width

        logger.debug("column widths %s", self.col_widths)

        self.format_str = "|".join(format_list)

        self.header_format_str = "+".join(header_sepa_format_list)
        self.header_data = {}
        for k in self.col_widths:
            self.header_data[k] = "-" * self.col_widths[k]