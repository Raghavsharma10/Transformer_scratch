def shutdown(self):
        """Disconnect all connections and end the loop

        :returns: None
        :rtype: None
        :raises: None
        """
        log.debug('Shutting down %s' % self)
        self.disconnect_all()
        self._looping.clear()