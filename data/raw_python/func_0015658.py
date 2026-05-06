def insert(self, parent, position, row=None):
        """insert(parent, position, row=None)

        :param parent:
            A valid :obj:`Gtk.TreeIter`, or :obj:`None`
        :type parent: :obj:`Gtk.TreeIter` or :obj:`None`

        :param position:
            position to insert the new row, or -1 for last
        :type position: :obj:`int`

        :param row: a list of values to apply to the newly inserted row or :obj:`None`
        :type row: [:obj:`object`] or :obj:`None`

        :returns: a :obj:`Gtk.TreeIter` pointing to the new row
        :rtype: :obj:`Gtk.TreeIter`

        Creates a new row at `position`.  If parent is not :obj:`None`, then
        the row will be made a child of `parent`.  Otherwise, the row will be
        created at the toplevel. If `position` is -1 or is larger than the
        number of rows at that level, then the new row will be inserted to the
        end of the list.

        The returned iterator will point to the newly inserted row. The row
        will be empty after this function is called if `row` is :obj:`None`.
        To fill in values, you need to call :obj:`Gtk.TreeStore.set`\\() or
        :obj:`Gtk.TreeStore.set_value`\\().

        If `row` isn't :obj:`None` it has to be a list of values which will be
        used to fill the row.
        """

        return self._do_insert(parent, position, row)