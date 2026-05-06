def create_question_dialog(self, text, second_text):
        """
        Function creates a question dialog with title text
        and second_text
        """
        dialog = self.create_message_dialog(
            text, buttons=Gtk.ButtonsType.YES_NO, icon=Gtk.MessageType.QUESTION
        )
        dialog.format_secondary_text(second_text)
        response = dialog.run()
        dialog.destroy()
        return response