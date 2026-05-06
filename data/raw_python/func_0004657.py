def _insert_row(self, i, index):
        """
        Insert a new row in the DataFrame.

        :param i: index location to insert
        :param index: index value to insert into the index list
        :return: nothing
        """
        if i == len(self._index):
            self._add_row(index)
        else:
            self._index.insert(i, index)
            for c in range(len(self._columns)):
                self._data[c].insert(i, None)