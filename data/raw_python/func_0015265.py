def _open_path_window(self):
        """
        Hides this window and opens path window.
        Passes all needed data and kwargs.
        """
        self.data['top_assistant'] = self.top_assistant
        self.data['current_main_assistant'] = self.get_current_main_assistant()
        self.data['kwargs'] = self.kwargs
        self.path_window.open_window(self.data)
        self.main_win.hide()