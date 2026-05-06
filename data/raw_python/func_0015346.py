def back_btn_clicked(self, widget, data=None):
        """
        Event for back button.
        This occurs in case of devassistant fail.
        """
        self.remove_link_button()
        self.run_window.hide()
        self.parent.path_window.path_window.show()