def add_seperator(self):
        """
        Add separator between labels in menu that called on right mouse click.
        """
        m_item = Gtk.SeparatorMenuItem()
        self.menu.append(m_item)
        self.menu.show_all()