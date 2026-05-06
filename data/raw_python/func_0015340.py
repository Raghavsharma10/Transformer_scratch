def disable_buttons(self):
        """
        Function disables buttons
        """
        self.main_btn.set_sensitive(False)
        self.back_btn.hide()
        self.info_label.set_label('<span color="#FFA500">In progress...</span>')
        self.disable_close_window()
        if self.link is not None:
            self.link.hide()