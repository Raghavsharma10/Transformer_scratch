def right_click_event_statusicon(self, icon, button, time):
        """
        It's just way how popup menu works in GTK. Don't ask me how it works.
        """

        def pos(menu, aicon):
            """Just return menu"""
            return Gtk.StatusIcon.position_menu(menu, aicon)

        self.menu.popup(None, None, pos, icon, button, time)