def update_devices(self, selection=None):
        """Determines the order, in which the |Node| and |Element| objects
        currently handled by the |HydPy| objects need to be processed during
        a simulation time step.  Optionally, a |Selection| object for defining
        new |Node| and |Element| objects can be passed."""
        if selection is not None:
            self.nodes = selection.nodes
            self.elements = selection.elements
        self._update_deviceorder()