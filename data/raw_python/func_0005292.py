def get_first_builder_window(builder):
    """Get the first toplevel widget in a Gtk.Builder hierarchy.

    This is mostly used for guessing purposes, and an explicit naming is
    always going to be a better situation.
    """
    for obj in builder.get_objects():
        if isinstance(obj, Gtk.Window):
            # first window
            return obj