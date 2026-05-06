def add_menu_item(self, command, title):
        """
        Add mouse right click menu item.
        Args:
          command (callable): function that will be called after left mouse click on title
          title (str): label that will be shown in menu
        """
        m_item = Gtk.MenuItem()
        m_item.set_label(title)
        m_item.connect('activate', command)
        self.menu.append(m_item)
        self.menu.show_all()