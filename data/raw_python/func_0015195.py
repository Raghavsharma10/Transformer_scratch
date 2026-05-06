def create_checkbox(self, name, margin=10):
        """
        Function creates a checkbox with his name
        """
        chk_btn = Gtk.CheckButton(name)
        chk_btn.set_margin_right(margin)
        return chk_btn