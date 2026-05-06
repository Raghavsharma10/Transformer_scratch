def connect(self):
        """
        connect
        """
        _logger.debug("Start connecting to broker")
        while True:
            try:
                self.client.connect(self.broker_host, self.broker_port,
                                    self.broker_keepalive)
                break
            except Exception:
                _logger.debug(
                    "Connect failed. wait %s sec" % self.connect_delay)
                sleep(self.connect_delay)

        self.client.loop_forever()