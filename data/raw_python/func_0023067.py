def view_changed(self):
        """ Called when this camera is changes its view. Also called
        when its associated with a viewbox.
        """
        if self._resetting:
            return  # don't update anything while resetting (are in set_range)
        if self._viewbox:
            # Set range if necessary
            if self._xlim is None:
                args = self._set_range_args or ()
                self.set_range(*args)
            # Store default state if we have not set it yet
            if self._default_state is None:
                self.set_default_state()
            # Do the actual update
            self._update_transform()