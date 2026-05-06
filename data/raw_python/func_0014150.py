def _list_of_dicts_to_column_headers(list_of_dicts):
        """
        Detects if all entries in an list of ``dict``'s have identical keys.
        Returns the keys if all keys are the same and ``None`` otherwise.

        Parameters
        ----------
        list_of_dicts : list
            List of dictionaries to test for identical keys.

        Returns
        -------
        list or None
            List of column headers if all dictionary posessed the same keys. Returns ``None`` otherwise.
        """

        if len(list_of_dicts) < 2 or not all(isinstance(item, dict) for item in list_of_dicts):
            return None

        column_headers = list_of_dicts[0].keys()
        for d in list_of_dicts[1:]:
            if len(d.keys()) != len(column_headers) or not all(header in d for header in column_headers):
                return None
        return column_headers