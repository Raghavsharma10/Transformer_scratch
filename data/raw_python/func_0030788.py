def verify_state(self):
        """ Verify if session was not yet opened. If it is, open it and call
        connections C{connectionMade} """
        # If we're in CONNECTING state - send 'o' message to the client
        if self.state == SESSION_STATE.CONNECTING:
            self.handler.send_pack(proto.CONNECT)

        # Call parent implementation
        super(Session, self).verify_state()