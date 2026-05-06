def get_iter(self, path):
        """
        :param path: the :obj:`Gtk.TreePath`-struct
        :type path: :obj:`Gtk.TreePath`

        :raises: :class:`ValueError` if `path` doesn't exist
        :returns: a :obj:`Gtk.TreeIter`
        :rtype: :obj:`Gtk.TreeIter`

        Returns an iterator pointing to `path`. If `path` does not exist
        :class:`ValueError` is raised.
        """

        path = self._coerce_path(path)
        success, aiter = super(TreeModel, self).get_iter(path)
        if not success:
            raise ValueError("invalid tree path '%s'" % path)
        return aiter