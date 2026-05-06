def connect_event_handlers(self):
        """Connects event handlers to the figure."""
        self.figure.canvas.mpl_connect('close_event', self.evt_release)
        self.figure.canvas.mpl_connect('pause_event', self.evt_toggle_pause)