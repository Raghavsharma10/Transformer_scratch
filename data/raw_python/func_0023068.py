def on_canvas_change(self, event):
        """Canvas change event handler

        Parameters
        ----------
        event : instance of Event
            The event.
        """
        # Connect key events from canvas to camera. 
        # TODO: canvas should keep track of a single node with keyboard focus.
        if event.old is not None:
            event.old.events.key_press.disconnect(self.viewbox_key_event)
            event.old.events.key_release.disconnect(self.viewbox_key_event)
        if event.new is not None:
            event.new.events.key_press.connect(self.viewbox_key_event)
            event.new.events.key_release.connect(self.viewbox_key_event)