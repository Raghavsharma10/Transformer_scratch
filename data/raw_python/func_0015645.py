def iter_next(self, iter):
        """
        :param iter: the :obj:`Gtk.TreeIter`-struct
        :type iter: :obj:`Gtk.TreeIter`

        :returns: a :obj:`Gtk.TreeIter` or :obj:`None`
        :rtype: :obj:`Gtk.TreeIter` or :obj:`None`

        Returns an iterator pointing to the node following `iter` at the
        current level.

        If there is no next `iter`, :obj:`None` is returned.
        """

        next_iter = iter.copy()
        success = super(TreeModel, self).iter_next(next_iter)
        if success:
            return next_iter