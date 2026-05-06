def iterchildren(self):
        """Returns a :obj:`Gtk.TreeModelRowIter` for the row's children"""

        child_iter = self.model.iter_children(self.iter)
        return TreeModelRowIter(self.model, child_iter)