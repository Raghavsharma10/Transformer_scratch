def pretty_print(self, printer: Optional[Printer] = None, align: int = ALIGN_CENTER, border: bool = False):
        """
        Pretty prints the table.

        :param printer: The printer to print with.
        :param align: The alignment of the cells(Table.ALIGN_CENTER/ALIGN_LEFT/ALIGN_RIGHT)
        :param border: Whether to add a border around the table
        """
        if printer is None:
            printer = get_printer()
        table_string = self._get_pretty_table(indent=printer.indents_sum, align=align, border=border).get_string()
        if table_string != '':
            first_line = table_string.splitlines()[0]
            first_line_length = len(first_line) - len(re.findall(Printer._ANSI_REGEXP, first_line)) * \
                Printer._ANSI_COLOR_LENGTH
            if self.title_align == self.ALIGN_CENTER:
                title = '{}{}'.format(' ' * (first_line_length // 2 - len(self.title) // 2), self.title)
            elif self.title_align == self.ALIGN_LEFT:
                title = self.title
            else:
                title = '{}{}'.format(' ' * (first_line_length - len(self.title)), self.title)
            printer.write_line(printer.YELLOW + title)
            # We split the table to lines in order to keep the indentation.
            printer.write_line(table_string)