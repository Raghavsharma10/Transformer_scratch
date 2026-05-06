def browse_clicked(self, widget, data=None):
        """
        Function sets the directory to entry
        """
        text = self.gui_helper.create_file_chooser_dialog("Please select directory", self.path_window)
        if text is not None:
            data.set_text(text)