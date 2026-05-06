def sort_index(self):
        """
        Sort the Series by the index. The sort modifies the Series inplace

        :return: nothing
        """
        sort = sorted_list_indexes(self._index)
        # sort index
        self._index = blist([self._index[x] for x in sort]) if self._blist else [self._index[x] for x in sort]
        # sort data
        self._data = blist([self._data[x] for x in sort]) if self._blist else [self._data[x] for x in sort]