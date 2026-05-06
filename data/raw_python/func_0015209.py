def create_message_dialog(self, text, buttons=Gtk.ButtonsType.CLOSE, icon=Gtk.MessageType.WARNING):
        """
        Function creates a message dialog with text
        and relevant buttons
        """
        dialog = Gtk.MessageDialog(None,
                                   Gtk.DialogFlags.DESTROY_WITH_PARENT,
                                   icon,
                                   buttons,
                                   text
        )
        return dialog