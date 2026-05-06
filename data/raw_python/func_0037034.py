def sub_menu(self):
        """Create daemon submenu
        """
        submenu = gtk.Menu()
        self.start = gtk.ImageMenuItem("Start")
        self.stop = gtk.ImageMenuItem("Stop")
        self.restart = gtk.ImageMenuItem("Restart")
        self.status = gtk.ImageMenuItem("Status")

        self.start.show()
        self.stop.show()
        self.restart.show()
        self.status.show()

        img_Start = gtk.image_new_from_stock(gtk.STOCK_MEDIA_PLAY,
                                             gtk.ICON_SIZE_MENU)
        img_Start.show()
        self.start.set_image(img_Start)

        img_Stop = gtk.image_new_from_stock(gtk.STOCK_STOP,
                                            gtk.ICON_SIZE_MENU)
        img_Stop.show()
        self.stop.set_image(img_Stop)

        img_Restart = gtk.image_new_from_stock(gtk.STOCK_REFRESH,
                                               gtk.ICON_SIZE_MENU)
        img_Restart.show()
        self.restart.set_image(img_Restart)

        img_Status = gtk.image_new_from_stock(gtk.STOCK_DIALOG_QUESTION,
                                              gtk.ICON_SIZE_MENU)
        img_Status.show()
        self.status.set_image(img_Status)

        submenu.append(self.start)
        submenu.append(self.stop)
        submenu.append(self.restart)
        submenu.append(self.status)

        self.daemon = gtk.ImageMenuItem("Daemon")
        self.img_daemon = gtk.image_new_from_stock(self.daemon_STOCK,
                                                   gtk.ICON_SIZE_MENU)
        self.img_daemon.show()
        self.daemon.set_submenu(submenu)