def prev_window(self, widget, data=None):
        """
        Function returns to Main Window
        """
        self.path_window.hide()
        self.parent.open_window(widget, self.data)