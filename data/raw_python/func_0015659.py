def insert_before(self, parent, sibling, row=None):
        """insert_before(parent, sibling, row=None)

        :param parent: A valid :obj:`Gtk.TreeIter`, or :obj:`None`
        :type parent: :obj:`Gtk.TreeIter` or :obj:`None`

        :param sibling: A valid :obj:`Gtk.TreeIter`, or :obj:`None`
        :type sibling: :obj:`Gtk.TreeIter` or :obj:`None`

        :param row: a list of values to apply to the newly inserted row or :obj:`None`
        :type row: [:obj:`object`] or :obj:`None`

        :returns: a :obj:`Gtk.TreeIter` pointing to the new row
        :rtype: :obj:`Gtk.TreeIter`

        Inserts a new row before `sibling`. If `sibling` is :obj:`None`, then
        the row will be appended to `parent` 's children. If `parent` and
        `sibling` are :obj:`None`, then the row will be appended to the
        toplevel.  If both `sibling` and `parent` are set, then `parent` must
        be the parent of `sibling`.  When `sibling` is set, `parent` is
        optional.

        The returned iterator will point to this new row. The row will be
        empty after this function is called if `row` is :obj:`None`.  To fill
        in values, you need to call :obj:`Gtk.TreeStore.set`\\() or
        :obj:`Gtk.TreeStore.set_value`\\().

        If `row` isn't :obj:`None` it has to be a list of values which will be
        used to fill the row.
        """

        treeiter = Gtk.TreeStore.insert_before(self, parent, sibling)

        if row is not None:
            self.set_row(treeiter, row)

        return treeiter