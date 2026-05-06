def start(self, reloading=False):
        """Called when the module is loaded.

        If the load is due to a reload of the module, then the 'reloading'
        argument will be set to True. By default, this method calls the
        controller's listen() for each event in the self.event_handlers dict.
        """
        for event in self.event_handlers:
            self.controller.listen(event)