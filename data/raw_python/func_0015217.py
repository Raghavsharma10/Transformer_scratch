def create_cell_renderer_combo(self, tree_view, title="title", assign=0, editable=False, model=None, function=None):
        """'
        Function creates a CellRendererCombo with title, model
        """
        renderer_combo = Gtk.CellRendererCombo()
        renderer_combo.set_property('editable', editable)
        if model:
            renderer_combo.set_property('model', model)
        if function:
            renderer_combo.connect("edited", function)
        renderer_combo.set_property("text-column", 0)
        renderer_combo.set_property("has-entry", False)
        column = Gtk.TreeViewColumn(title, renderer_combo, text=assign)
        tree_view.append_column(column)