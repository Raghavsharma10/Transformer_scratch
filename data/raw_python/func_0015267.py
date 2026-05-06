def get_current_main_assistant(self):
        """
        Function return current assistant
        """
        current_page = self.notebook.get_nth_page(self.notebook.get_current_page())
        return current_page.main_assistant