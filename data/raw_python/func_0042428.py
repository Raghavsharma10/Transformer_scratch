def get_as_html(self) -> str:
        """
        Returns the table object as an HTML string.

        :return: HTML representation of the table.
        """
        table_string = self._get_pretty_table().get_html_string()
        title = ('{:^' + str(len(table_string.splitlines()[0])) + '}').format(self.title)
        return f'<center><h1>{title}</h1></center>{table_string}'