def delete_all_rows(self):
        """
        Deletes the contents of all rows in the DataFrame. This function is faster than delete_rows() to remove all
        information, and at the same time it keeps the container lists for the columns and index so if there is another
        object that references this DataFrame, like a ViewSeries, the reference remains in tact.
        
        :return: nothing
        """
        del self._index[:]
        for c in range(len(self._columns)):
            del self._data[c][:]