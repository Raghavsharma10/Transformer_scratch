def _auto_send(self):
        """
        auto send blocking function, when the interval or the message size has been reached, publish
        :return:
        """
        while True:
            if time.time() - self.last_send_time > self.config.async_auto_send_interval_millis or \
                            len(self._tx_queue) >= self.config.async_auto_send_amount:
                self.publish_queue()