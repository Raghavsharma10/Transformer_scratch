def _check_box_toggled(self, widget, data=None):
        """
        Function manipulates with entries and buttons.
        """
        active = widget.get_active()
        arg_name = data

        if 'entry' in self.args[arg_name]:
            self.args[arg_name]['entry'].set_sensitive(active)
        if 'browse_btn' in self.args[arg_name]:
            self.args[arg_name]['browse_btn'].set_sensitive(active)

        self.path_window.show_all()