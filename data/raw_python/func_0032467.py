def switchTo(self, app):
        """
        Use the given L{ITerminalServerFactory} to create a new
        L{ITerminalProtocol} and connect it to C{self.terminal} (such that it
        cannot actually disconnect, but can do most anything else).  Control of
        the terminal is delegated to it until it gives up that control by
        disconnecting itself from the terminal.

        @type app: L{ITerminalServerFactory} provider
        @param app: The factory which will be used to create a protocol
            instance.
        """
        viewer = _AuthenticatedShellViewer(list(getAccountNames(self._store)))
        self._protocol = app.buildTerminalProtocol(viewer)
        self._protocol.makeConnection(_ReturnToMenuWrapper(self, self.terminal))