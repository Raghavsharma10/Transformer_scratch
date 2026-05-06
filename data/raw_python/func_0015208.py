def create_notebook(self, position=Gtk.PositionType.TOP):
        """
        Function creates a notebook
        """
        notebook = Gtk.Notebook()
        notebook.set_tab_pos(position)
        notebook.set_show_border(True)
        return notebook