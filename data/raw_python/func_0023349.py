def on_resize(self, event):
        """Resize handler

        Parameters
        ----------
        event : instance of Event
            The resize event.
        """
        self._update_transforms()
        
        if self._central_widget is not None:
            self._central_widget.size = self.size
            
        if len(self._vp_stack) == 0:
            self.context.set_viewport(0, 0, *self.physical_size)