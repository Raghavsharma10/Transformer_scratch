def show_and_run(self):
        """Show the main widget in a window and run the gtk loop"""
        if not self._ui_ready:
            self.prepare_ui()
        self.display_widget = Gtk.Window()
        self.display_widget.add(self.widget)
        self.display_widget.show()
        self.display_widget.connect('destroy', lambda *args: self.hide_and_quit())
        BaseDelegate.show_and_run(self)