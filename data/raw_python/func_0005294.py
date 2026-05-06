def get_builder_toplevel(self, builder):
        """Get the toplevel widget from a Gtk.Builder file.

        The slave view implementation first searches for the widget named as
        self.toplevel_name (which defaults to "main". If this is missing, the
        first toplevel widget is discovered in the Builder file, and it's
        immediate child is used as the toplevel widget for the delegate.
        """
        toplevel = builder.get_object(self.toplevel_name)
        if toplevel is None:
            toplevel = get_first_builder_window(builder).child
        if toplevel is not None:
            #XXX: what to do if a developer
            #     gave the name of a window instead of its child
            toplevel.get_parent().remove(toplevel)
        return toplevel