def menu(self, event_button, event_time, data=None):
        """Create popup menu
        """
        self.sub_menu()
        menu = gtk.Menu()
        menu.append(self.daemon)

        separator = gtk.SeparatorMenuItem()

        menu_Check = gtk.ImageMenuItem("Check updates")
        img_Check = gtk.image_new_from_stock(gtk.STOCK_OK,
                                             gtk.ICON_SIZE_MENU)
        img_Check.show()

        menu_Info = gtk.ImageMenuItem("OS Info")
        img_Info = gtk.image_new_from_stock(gtk.STOCK_INFO,
                                            gtk.ICON_SIZE_MENU)
        img_Info.show()

        menu_About = gtk.ImageMenuItem("About")
        img_About = gtk.image_new_from_stock(gtk.STOCK_ABOUT,
                                             gtk.ICON_SIZE_MENU)
        img_About.show()

        self.daemon.set_image(self.img_daemon)
        menu.append(self.daemon)
        self.daemon.show()

        menu_Quit = gtk.ImageMenuItem("Quit")
        img_Quit = gtk.image_new_from_stock(gtk.STOCK_QUIT, gtk.ICON_SIZE_MENU)
        img_Quit.show()

        menu_Check.set_image(img_Check)
        menu_Info.set_image(img_Info)
        menu_About.set_image(img_About)
        menu_Quit.set_image(img_Quit)

        menu.append(menu_Check)
        menu.append(menu_Info)
        menu.append(separator)
        menu.append(menu_About)
        menu.append(menu_Quit)

        separator.show()
        menu_Check.show()
        menu_Info.show()
        menu_About.show()
        menu_Quit.show()

        menu_Check.connect_object("activate", self._Check, " ")
        menu_Info.connect_object("activate", self._Info, "OS Info")
        menu_About.connect_object("activate", self._About, "SUN")
        self.start.connect_object("activate", self._start, "Start daemon   ")
        self.stop.connect_object("activate", self._stop, "Stop daemon   ")
        self.restart.connect_object("activate", self._restart,
                                    "Restart daemon   ")
        self.status.connect_object("activate", self._status, daemon_status())
        menu_Quit.connect_object("activate", self._Quit, "stop")

        menu.popup(None, None, None, event_button, event_time, data)