def allow_buttons(self, message="", link=True, back=True):
        """
        Function allows buttons
        """
        self.info_label.set_label(message)
        self.allow_close_window()
        if link and self.link is not None:
            self.link.set_sensitive(True)
            self.link.show_all()
        if back:
            self.back_btn.show()
        self.main_btn.set_sensitive(True)