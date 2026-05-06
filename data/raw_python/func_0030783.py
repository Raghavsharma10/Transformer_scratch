def delayed_close(self):
        """ Delayed close - won't close immediately, but on the next reactor
        loop. """
        self.state = SESSION_STATE.CLOSING
        reactor.callLater(0, self.close)