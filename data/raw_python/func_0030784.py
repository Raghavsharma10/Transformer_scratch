def is_closed(self):
        """ Check if session was closed. """
        return (self.state == SESSION_STATE.CLOSED 
                or self.state == SESSION_STATE.CLOSING)