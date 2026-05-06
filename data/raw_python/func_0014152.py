def _maybe_club(self, list_of_dicts):
        """
        If all keys in a list of dicts are identical, values from each ``dict`` 
        are clubbed, i.e. inserted under a common column heading. If the keys
        are not identical ``None`` is returned, and the list should be converted
        to HTML per the normal ``convert`` function.

        Parameters
        ----------
        list_of_dicts : list
            List to attempt to club.

        Returns
        -------
        str or None
            String of HTML if list was successfully clubbed. Returns ``None`` otherwise.

        Example
        -------
        Given the following json object::

            {
                "sampleData": [
                    {"a":1, "b":2, "c":3},
                    {"a":5, "b":6, "c":7}]
            }

                
        Calling ``_maybe_club`` would result in the following HTML table:
        _____________________________
        |               |   |   |   |
        |               | a | c | b |
        |   sampleData  |---|---|---|
        |               | 1 | 3 | 2 |
        |               | 5 | 7 | 6 |
        -----------------------------

        Adapted from a contribution from @muellermichel to ``json2html``.
        """
        column_headers = JsonConverter._list_of_dicts_to_column_headers(list_of_dicts)
        if column_headers is None:
            # common headers not found, return normal markup
            html_output = self._markup(list_of_dicts)
        else:
            html_output = self._table_opening_tag
            html_output += self._markup_header_row(column_headers)
            for list_entry in list_of_dicts:
                html_output += "<tr><td>"
                html_output += "</td><td>".join(self._markup(list_entry[column_header]) for column_header in column_headers)
                html_output += "</td></tr>"
            html_output += "</table>"

        return self._markup_table_cell(html_output)