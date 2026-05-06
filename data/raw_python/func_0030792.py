def close(self, code=3000, message='Go away!'):
        """ Close session.

        @param code: Closing code

        @param message: Closing message
        """
        if self.state != SESSION_STATE.CLOSED:
            # Notify handler
            if self.handler is not None:
                self.handler.send_pack(proto.disconnect(code, message))

        super(Session, self).close(code, message)