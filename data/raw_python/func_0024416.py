def on_message(self, message):
        """
        called on new websocket message,
        """
        log.debug("WS MSG for %s: %s" % (self._get_sess_id(), message))
        self.application.pc.redirect_incoming_message(self._get_sess_id(), message, self.request)