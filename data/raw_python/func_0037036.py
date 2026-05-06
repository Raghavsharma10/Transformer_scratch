def message(self, data):
        """Function to display messages to the user
        """
        msg = gtk.MessageDialog(None, gtk.DIALOG_MODAL, gtk.MESSAGE_INFO,
                                gtk.BUTTONS_CLOSE, data)
        msg.set_resizable(1)
        msg.set_title(self.dialog_title)
        self.img.set_from_file(self.sun_icon)
        msg.set_image(self.img)
        msg.show_all()
        msg.run()
        msg.destroy()