def _markup(self, entry):
        """
        Recursively generates HTML for the current entry.

        Parameters
        ----------
        entry : object
            Object to convert to HTML. Maybe be a single entity or contain multiple and/or nested objects.

        Returns
        -------
        str
            String of HTML formatted json.
        """
        if entry is None:
            return ""
        if isinstance(entry, list):
            list_markup = "<ul>"
            for item in entry:
                list_markup += "<li>{:s}</li>".format(self._markup(item))
            list_markup += "</ul>"
            return list_markup
        if isinstance(entry, dict):
            return self.convert(entry)

        # default to stringifying entry
        return str(entry)