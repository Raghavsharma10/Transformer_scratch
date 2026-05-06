def execute_dialog(self, title):
        """
        Function executes a dialog
        """
        msg_dlg = self.create_message_dialog(title)
        msg_dlg.run()
        msg_dlg.destroy()
        return