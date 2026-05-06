def reactivate(self):
        """
        Called when a sub-protocol is finished.  This disconnects the
        sub-protocol and redraws the main menu UI.
        """
        self._protocol.connectionLost(None)
        self._protocol = None
        self.terminal.reset()
        self._window.filthy()
        self._window.repaint()