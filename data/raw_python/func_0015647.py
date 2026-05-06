def set_row(self, treeiter, row):
        """
        :param treeiter: the :obj:`Gtk.TreeIter`
        :type treeiter: :obj:`Gtk.TreeIter`

        :param row: a list of values for each column
        :type row: [:obj:`object`]

        Sets all values of a row pointed to by `treeiter` from a list of
        values passes as `row`. The length of the row has to match the number
        of columns of the model. :obj:`None` in `row` means the value will be
        skipped and not set.

        Also see :obj:`Gtk.ListStore.set_value`\\() and
        :obj:`Gtk.TreeStore.set_value`\\()
        """

        converted_row, columns = self._convert_row(row)
        for column in columns:
            value = row[column]
            if value is None:
                continue  # None means skip this row

            self.set_value(treeiter, column, value)