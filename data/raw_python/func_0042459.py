def sync(self, raw_data, row_change_callback=None):
        """ Equivalent to the inject method but will delete rows from the
        google spreadsheet if their key is not found in the input (raw_data) 
        dictionary.
    
        Args:
            raw_data (dict): See inject method
            row_change_callback (Optional) (func): See inject method

        Returns:
            UpdateResults (object): See inject method
        """
        return self._update(raw_data, row_change_callback, delete_rows=True)