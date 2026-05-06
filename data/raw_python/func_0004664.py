def set_location(self, location, values, missing_to_none=False):
        """
        Sets the column values, as given by the keys of the values dict, for the row at location to the values of the
        values dict. If missing_to_none is False then columns not in the values dict will be left unchanged, if it is
        True then they are set to None. This method does not add new columns and raises an error.
        
        :param location: location
        :param values: dict of column names as keys and the value as the value to set the row for that column to 
        :param missing_to_none: if True set any column missing in the values to None, otherwise leave unchanged
        :return: nothing
        """
        if missing_to_none:
            # populate the dict with None in any column missing
            for column in self._columns:
                if column not in values:
                    values[column] = None

        for column in values:
            i = self._columns.index(column)
            self._data[i][location] = values[column]