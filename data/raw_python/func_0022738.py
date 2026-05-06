def viewbox_key_event(self, event):
        """ViewBox key event handler

        Parameters
        ----------
        event : instance of Event
            The event.
        """
        PerspectiveCamera.viewbox_key_event(self, event)

        if event.handled or not self.interactive:
            return

        # Ensure the timer runs
        if not self._timer.running:
            self._timer.start()

        if event.key in self._keymap:
            val_dims = self._keymap[event.key]
            val = val_dims[0]
            # Brake or accelarate?
            if val == 0:
                vec = self._brake
                val = 1
            else:
                vec = self._acc
            # Set
            if event.type == 'key_release':
                val = 0
            for dim in val_dims[1:]:
                factor = 1.0
                vec[dim-1] = val * factor