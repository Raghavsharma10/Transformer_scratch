def sort_index(self):
        """
        Sort the DataFrame by the index. The sort modifies the DataFrame inplace

        :return: nothing
        """
        sort = sorted_list_indexes(self._index)
        # sort index
        self._index = blist([self._index[x] for x in sort]) if self._blist else [self._index[x] for x in sort]
        # each column
        for c in range(len(self._data)):
            self._data[c] = blist([self._data[c][i] for i in sort]) if self._blist else [self._data[c][i] for i in sort]