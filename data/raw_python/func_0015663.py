def get_selected(self):
        """
        :returns:
            :model: the :obj:`Gtk.TreeModel`
            :iter: The :obj:`Gtk.TreeIter` or :obj:`None`

        :rtype: (**model**: :obj:`Gtk.TreeModel`, **iter**: :obj:`Gtk.TreeIter` or :obj:`None`)

        {{ docs }}
        """

        success, model, aiter = super(TreeSelection, self).get_selected()
        if success:
            return (model, aiter)
        else:
            return (model, None)