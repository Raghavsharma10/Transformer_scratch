def _log_disconnect(self):
        """ Decrement connection count """
        if self.logged:
            self.server.stats.connectionClosed()
            self.logged = False