def browse_path(self, window):
        """
        Function opens the file chooser dialog for settings project dir
        """
        text = self.gui_helper.create_file_chooser_dialog("Choose project directory", self.path_window, name="Select")
        if text is not None:
            self.dir_name.set_text(text)