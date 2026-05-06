def create_file_chooser_dialog(self, text, parent, name=Gtk.STOCK_OPEN):
        """
        Function creates a file chooser dialog with title text
        """
        text = None
        dialog = Gtk.FileChooserDialog(
            text, parent,
            Gtk.FileChooserAction.SELECT_FOLDER,
            (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, name, Gtk.ResponseType.OK)
        )
        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            text = dialog.get_filename()
        dialog.destroy()
        return text