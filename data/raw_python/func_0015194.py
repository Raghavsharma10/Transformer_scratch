def checkbutton_with_label(self, description):
        """
            The function creates a checkbutton with label
        """
        act_btn = Gtk.CheckButton(description)
        align = self.create_alignment()
        act_btn.add(align)
        return align