def insert_before(self, sibling, row=None):
        """insert_before(sibling, row=None)

        :param sibling: A valid :obj:`Gtk.TreeIter`, or :obj:`None`
        :type sibling: :obj:`Gtk.TreeIter` or :obj:`None`

        :param row: a list of values to apply to the newly inserted row or :obj:`None`
        :type row: [:obj:`object`] or :obj:`None`

        :returns: :obj:`Gtk.TreeIter` pointing to the new row
        :rtype: :obj:`Gtk.TreeIter`

        Inserts a new row before `sibling`. If `sibling` is :obj:`None`, then
        the row will be appended to the end of the list.

        The row will be empty if `row` is :obj:`None. To fill in values, you
        need to call :obj:`Gtk.ListStore.set`\\() or
        :obj:`Gtk.ListStore.set_value`\\().

        If `row` isn't :obj:`None` it has to be a list of values which will be
        used to fill the row.
        """

        treeiter = Gtk.ListStore.insert_before(self, sibling)

        if row is not None:
            self.set_row(treeiter, row)

        return treeiter