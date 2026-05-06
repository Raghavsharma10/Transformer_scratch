def create_clipboard(self, text, selection=Gdk.SELECTION_CLIPBOARD):
        """
        Function creates a clipboard
        """
        clipboard = Gtk.Clipboard.get(selection)
        clipboard.set_text('\n'.join(text), -1)
        clipboard.store()
        return clipboard