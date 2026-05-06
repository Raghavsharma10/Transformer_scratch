def get_selected_rows(self):
        """
        :returns:
            A list containing a :obj:`Gtk.TreePath` for each selected row
            and a :obj:`Gtk.TreeModel` or :obj:`None`.

        :rtype: (:obj:`Gtk.TreeModel`, [:obj:`Gtk.TreePath`])

        {{ docs }}
        """

        rows, model = super(TreeSelection, self).get_selected_rows()
        return (model, rows)