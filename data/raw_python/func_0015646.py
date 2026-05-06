def iter_previous(self, iter):
        """
        :param iter: the :obj:`Gtk.TreeIter`-struct
        :type iter: :obj:`Gtk.TreeIter`

        :returns: a :obj:`Gtk.TreeIter` or :obj:`None`
        :rtype: :obj:`Gtk.TreeIter` or :obj:`None`

        Returns an iterator pointing to the previous node at the current level.

        If there is no previous `iter`, :obj:`None` is returned.
        """

        prev_iter = iter.copy()
        success = super(TreeModel, self).iter_previous(prev_iter)
        if success:
            return prev_iter