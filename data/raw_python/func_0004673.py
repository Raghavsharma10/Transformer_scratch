def sort_columns(self, column, key=None, reverse=False):
        """
        Sort the DataFrame by one of the columns. The sort modifies the DataFrame inplace. The key and reverse
        parameters have the same meaning as for the built-in sort() function.

        :param column: column name to use for the sort
        :param key: if not None then a function of one argument that is used to extract a comparison key from each
                    list element
        :param reverse: if True then the list elements are sort as if each comparison were reversed.
        :return: nothing
        """
        if isinstance(column, (list, blist)):
            raise TypeError('Can only sort by a single column  ')
        sort = sorted_list_indexes(self._data[self._columns.index(column)], key, reverse)
        # sort index
        self._index = blist([self._index[x] for x in sort]) if self._blist else [self._index[x] for x in sort]
        # each column
        for c in range(len(self._data)):
            self._data[c] = blist([self._data[c][i] for i in sort]) if self._blist else [self._data[c][i] for i in sort]