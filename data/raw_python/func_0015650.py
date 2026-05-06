def append(self, row=None):
        """append(row=None)

        :param row: a list of values to apply to the newly append row or :obj:`None`
        :type row: [:obj:`object`] or :obj:`None`

        :returns: :obj:`Gtk.TreeIter` of the appended row
        :rtype: :obj:`Gtk.TreeIter`

        If `row` is :obj:`None` the appended row will be empty and to fill in
        values you need to call :obj:`Gtk.ListStore.set`\\() or
        :obj:`Gtk.ListStore.set_value`\\().

        If `row` isn't :obj:`None` it has to be a list of values which will be
        used to fill the row .
        """

        if row:
            return self._do_insert(-1, row)
        # gtk_list_store_insert() does not know about the "position == -1"
        # case, so use append() here
        else:
            return Gtk.ListStore.append(self)