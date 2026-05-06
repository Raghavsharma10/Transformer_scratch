def get_previous(self):
        """Returns the previous :obj:`Gtk.TreeModelRow` or None"""

        prev_iter = self.model.iter_previous(self.iter)
        if prev_iter:
            return TreeModelRow(self.model, prev_iter)