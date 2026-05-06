def _switchTo(self, newProto, clientFactory=None):
        """ Switch this Juice instance to a new protocol.  You need to do this
        'simultaneously' on both ends of a connection; the easiest way to do
        this is to use a subclass of ProtocolSwitchCommand.
        """

        assert self.innerProtocol is None, "Protocol can only be safely switched once."
        self.setRawMode()
        self.innerProtocol = newProto
        self.innerProtocolClientFactory = clientFactory
        newProto.makeConnection(self.transport)