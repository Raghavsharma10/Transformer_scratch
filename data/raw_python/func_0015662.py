def set_attributes(self, cell_renderer, **attributes):
        """
        :param cell_renderer: the :obj:`Gtk.CellRenderer` we're setting the attributes of
        :type cell_renderer: :obj:`Gtk.CellRenderer`

        {{ docs }}
        """

        Gtk.CellLayout.clear_attributes(self, cell_renderer)

        for (name, value) in attributes.items():
            Gtk.CellLayout.add_attribute(self, cell_renderer, name, value)