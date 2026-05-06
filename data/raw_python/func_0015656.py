def get_parent(self):
        """Returns the parent :obj:`Gtk.TreeModelRow` or htis row or None"""

        parent_iter = self.model.iter_parent(self.iter)
        if parent_iter:
            return TreeModelRow(self.model, parent_iter)