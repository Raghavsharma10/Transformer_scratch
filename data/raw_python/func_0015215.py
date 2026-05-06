def create_tree_view(self, model=None):
        """
        Function creates a tree_view with model
        """
        tree_view = Gtk.TreeView()
        if model is not None:
            tree_view.set_model(model)
        return tree_view