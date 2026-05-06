def create_cell_renderer_text(self, tree_view, title="title", assign=0, editable=False):
        """
        Function creates a CellRendererText with title
        """
        renderer = Gtk.CellRendererText()
        renderer.set_property('editable', editable)
        column = Gtk.TreeViewColumn(title, renderer, text=assign)
        tree_view.append_column(column)