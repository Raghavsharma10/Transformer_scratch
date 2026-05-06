def button_with_label(self, description, assistants=None):
        """
            Function creates a button with lave.
            If assistant is specified then text is aligned
        """
        btn = self.create_button()
        label = self.create_label(description)
        if assistants is not None:
            h_box = self.create_box(orientation=Gtk.Orientation.VERTICAL)
            h_box.pack_start(label, False, False, 0)
            label_ass = self.create_label(
                assistants, justify=Gtk.Justification.LEFT
            )
            label_ass.set_alignment(0, 0)
            h_box.pack_start(label_ass, False, False, 12)
            btn.add(h_box)
        else:
            btn.add(label)
        return btn