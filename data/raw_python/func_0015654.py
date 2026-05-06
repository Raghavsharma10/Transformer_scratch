def get_next(self):
        """Returns the next :obj:`Gtk.TreeModelRow` or None"""

        next_iter = self.model.iter_next(self.iter)
        if next_iter:
            return TreeModelRow(self.model, next_iter)