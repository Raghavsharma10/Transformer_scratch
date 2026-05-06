def get_builder_toplevel(self, builder):
        """Get the toplevel widget from a Gtk.Builder file.

        The main view implementation first searches for the widget named as
        self.toplevel_name (which defaults to "main". If this is missing, or not
        a Gtk.Window, the first toplevel window found in the Gtk.Builder is
        used.
        """
        toplevel = builder.get_object(self.toplevel_name)
        if not GObject.type_is_a(toplevel, Gtk.Window):
            toplevel = None
        if toplevel is None:
            toplevel = get_first_builder_window(builder)
        return toplevel