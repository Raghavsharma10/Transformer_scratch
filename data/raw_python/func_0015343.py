def debug_btn_clicked(self, widget, data=None):
        """
        Event in case that debug button is pressed.
        """
        self.store.clear()
        self.thread = threading.Thread(target=self.logs_update)
        self.thread.start()