def create_entry(self, text="", sensitive="False"):
        """
        Function creates an Entry with corresponding text
        """
        text_entry = Gtk.Entry()
        text_entry.set_sensitive(sensitive)
        text_entry.set_text(text)
        return text_entry