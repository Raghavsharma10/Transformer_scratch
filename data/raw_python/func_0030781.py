def verify_state(self):
        """ Verify if session was not yet opened. If it is, open it and call
        connection's C{connectionMade} """
        if self.state == SESSION_STATE.CONNECTING:
            self.state = SESSION_STATE.OPEN

            self.conn.connectionMade(self.conn_info)