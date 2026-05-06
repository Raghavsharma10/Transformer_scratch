def reset_index(self, drop=False):
        """
        Resets the index of the DataFrame to simple integer list and the index name to 'index'. If drop is True then
        the existing index is dropped, if drop is False then the current index is made a column in the DataFrame with
        the index name the name of the column. If the index is a tuple multi-index then each element of the tuple is
        converted into a separate column. If the index name was 'index' then the column name will be 'index_0' to not
        conflict on print().

        :param drop: if True then the current index is dropped, if False then index converted to columns
        :return: nothing
        """
        if not drop:
            if isinstance(self.index_name, tuple):
                index_data = list(map(list, zip(*self._index)))
                for i in range(len(self.index_name)):
                    self.set_column(column=self.index_name[i], values=index_data[i])
            else:
                col_name = self.index_name if self.index_name is not 'index' else 'index_0'
                self.set_column(column=col_name, values=self._index)
        self.index = list(range(self.__len__()))
        self.index_name = 'index'