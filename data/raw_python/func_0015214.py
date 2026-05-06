def create_textview(self, wrap_mode=Gtk.WrapMode.WORD_CHAR, justify=Gtk.Justification.LEFT, visible=True, editable=True):
        """
        Function creates a text view with wrap_mode
        and justification
        """
        text_view = Gtk.TextView()
        text_view.set_wrap_mode(wrap_mode)
        text_view.set_editable(editable)
        if not editable:
            text_view.set_cursor_visible(False)
        else:
            text_view.set_cursor_visible(visible)
        text_view.set_justification(justify)
        return text_view