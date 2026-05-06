def keystrokeReceived(self, keyID, modifier):
        """
        Forward input events to the application-supplied protocol if one is
        currently active, otherwise forward them to the main menu UI.
        """
        if self._protocol is not None:
            self._protocol.keystrokeReceived(keyID, modifier)
        else:
            self._window.keystrokeReceived(keyID, modifier)