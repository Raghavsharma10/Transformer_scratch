def hook_changed(self, hook_name, widget, new_data):
        """Handle a hook upate."""
        if hook_name == 'song':
            self.song_changed(widget, new_data)
        elif hook_name == 'state':
            self.state_changed(widget, new_data)
        elif hook_name == 'elapsed_and_total':
            elapsed, total = new_data
            self.time_changed(widget, elapsed, total)