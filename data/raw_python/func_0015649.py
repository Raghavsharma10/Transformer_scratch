def get(self, treeiter, *columns):
        """
        :param treeiter: the :obj:`Gtk.TreeIter`
        :type treeiter: :obj:`Gtk.TreeIter`

        :param \\*columns: a list of column indices to fetch
        :type columns: (:obj:`int`)

        Returns a tuple of all values specified by their indices in `columns`
        in the order the indices are contained in `columns`

        Also see :obj:`Gtk.TreeStore.get_value`\\()
        """

        n_columns = self.get_n_columns()

        values = []
        for col in columns:
            if not isinstance(col, int):
                raise TypeError("column numbers must be ints")

            if col < 0 or col >= n_columns:
                raise ValueError("column number is out of range")

            values.append(self.get_value(treeiter, col))

        return tuple(values)