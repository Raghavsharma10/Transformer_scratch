def inject(self, raw_data, row_change_callback=None):
        """ Use this function to add rows or update existing rows in the
        spreadsheet.
    
        Args: 
          raw_data (dict): A dictionary of dictionaries. Where the keys of the
             outer dictionary uniquely identify each row of data, and the inner
             dictionaries represent the field,value pairs for a row of data.
   
          row_change_callback (Optional) (func): A callback function that you
             can use to track changes to rows on the spreadsheet. The
             row_change_callback function must take four parameters like so:

             change_callback(row_key, 
                             row_dict_before, 
                             row_dict_after, 
                             list_of_changed_keys)

        Returns:
          UpdateResults (object): A simple counter object providing statistics
            about the changes made by sheetsync.
        """
        return self._update(raw_data, row_change_callback, delete_rows=False)