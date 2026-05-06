def insert_column_with_attributes(self, position, title, cell, **kwargs):
        """
        :param position: The position to insert the new column in
        :type position: :obj:`int`

        :param title: The title to set the header to
        :type title: :obj:`str`

        :param cell: The :obj:`Gtk.CellRenderer`
        :type cell: :obj:`Gtk.CellRenderer`

        {{ docs }}
        """

        column = TreeViewColumn()
        column.set_title(title)
        column.pack_start(cell, False)
        self.insert_column(column, position)
        column.set_attributes(cell, **kwargs)