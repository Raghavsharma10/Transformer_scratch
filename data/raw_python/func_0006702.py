def set_interface(self, interface):
        """Add update toolbar callback to the interface"""
        self.interface = interface
        self.interface.callbacks.update_toolbar = self._update
        self._update()