def do_startup(self):
        """Gtk.Application.run() will call this function()"""

        Gtk.Application.do_startup(self)
        gtk_window = Gtk.ApplicationWindow(application=self)
        gtk_window.set_title('AppKit')
        webkit_web_view = WebKit.WebView()
        webkit_web_view.load_uri('http://localhost:' + str(self.port))

        screen = Gdk.Screen.get_default()
        monitor_geometry = screen.get_primary_monitor()
        monitor_geometry = screen.get_monitor_geometry(monitor_geometry)

        settings = webkit_web_view.get_settings()
        settings.set_property('enable-universal-access-from-file-uris', True)
        settings.set_property('enable-file-access-from-file-uris', True)
        settings.set_property('default-encoding', 'utf-8')
        gtk_window.set_default_size(
            monitor_geometry.width * 1.0 / 2.0,
            monitor_geometry.height * 3.0 / 5.0,
        )
        scrollWindow = Gtk.ScrolledWindow()
        scrollWindow.add(webkit_web_view)
        gtk_window.add(scrollWindow)
        gtk_window.connect('delete-event', self._on_gtk_window_destroy)
        webkit_web_view.connect('notify::title', self._on_notify_title)
        self.gtk_window = gtk_window
        self.webkit_web_view = webkit_web_view
        gtk_window.show_all()