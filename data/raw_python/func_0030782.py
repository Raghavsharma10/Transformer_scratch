def close(self, code=3000, message='Go away!'):
        """ Close session or endpoint connection.

        @param code: Closing code

        @param message: Close message
        """
        if self.state != SESSION_STATE.CLOSED:
            try:
                self.conn.connectionLost()
            except Exception as e:
                log.msg("Failed to call connectionLost(): %r." % e)
            finally:
                self.state = SESSION_STATE.CLOSED
                self.close_reason = (code, message)

            # Bump stats
            self.stats.sessionClosed(self.transport_name)

            # If we have active handler, notify that session was closed
            if self.handler is not None:
                self.handler.session_closed()