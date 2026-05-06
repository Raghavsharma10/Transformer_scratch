def _get_pretty_table(self, indent: int = 0, align: int = ALIGN_CENTER, border: bool = False) -> PrettyTable:
        """
        Returns the table format of the scheme, i.e.:

            <table name>
        +----------------+----------------
        |    <field1>    |   <field2>...
        +----------------+----------------
        | value1(field1) |  value1(field2)
        | value2(field1) |  value2(field2)
        | value3(field1) |  value3(field2)
        +----------------+----------------
        """
        rows = self.rows
        columns = self.columns
        # Add the column color.
        if self._headers_color != Printer.NORMAL and len(rows) > 0 and len(columns) > 0:
            # We need to copy the lists so that we wont insert colors in the original ones.
            rows[0] = rows[0][:]
            columns = columns[:]
            columns[0] = self._headers_color + columns[0]
            # Write the table itself in NORMAL color.
            rows[0][0] = Printer.NORMAL + str(rows[0][0])

        table = PrettyTable(columns, border=border, max_width=get_console_width() - indent)
        table.align = self._ALIGN_DICTIONARY[align]

        for row in rows:
            table.add_row(row)

        # Set the max width according to the columns size dict, or by default size limit when columns were not provided.
        for column, max_width in self._column_size_map.items():
            table.max_width[column] = max_width

        return table