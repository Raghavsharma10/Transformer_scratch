def create_button(self, style=Gtk.ReliefStyle.NORMAL):
        """
        This is generalized method for creating Gtk.Button
        """
        btn = Gtk.Button()
        btn.set_relief(style)
        return btn