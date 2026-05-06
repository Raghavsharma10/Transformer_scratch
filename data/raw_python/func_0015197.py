def create_link_button(self, text="None", uri="None"):
        """
        Function creates a link button with corresponding text and
        URI reference
        """
        link_btn = Gtk.LinkButton(uri, text)
        return link_btn