def switch_cursor(cursor_type, parent_window):
    """
    Functions switches the cursor to cursor type
    """
    watch = Gdk.Cursor(cursor_type)
    window = parent_window.get_root_window()
    window.set_cursor(watch)