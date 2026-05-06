def convert(self, json_input):
        """
        Converts JSON to HTML Table format.

        Parameters
        ----------
        json_input : dict
            JSON object to convert into HTML.

        Returns
        -------
        str
            String of converted HTML.
        """
        html_output = self._table_opening_tag
        if self._build_top_to_bottom:
            html_output += self._markup_header_row(json_input.keys())
            html_output += "<tr>"
            for value in json_input.values():
                if isinstance(value, list):
                    # check if all keys in the list are identical
                    # and group all values under a common column
                    # heading if so, if not default to normal markup
                    html_output += self._maybe_club(value)
                else:
                    html_output += self._markup_table_cell(value)
            html_output += "</tr>"
        else:
            for key, value in iter(json_input.items()):
                html_output += "<tr><th>{:s}</th>".format(self._markup(key))
                if isinstance(value, list):
                    html_output += self._maybe_club(value)
                else:
                    html_output += self._markup_table_cell(value)
                html_output += "</tr>"
        html_output += "</table>"
        return html_output