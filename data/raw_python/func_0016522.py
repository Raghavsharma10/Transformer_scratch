def consume(self, no_ack=None):
        """Declare consumer."""
        no_ack = no_ack or self.no_ack
        self.backend.declare_consumer(queue=self.queue, no_ack=no_ack,
                                      callback=self._receive_callback,
                                      consumer_tag=self.consumer_tag,
                                      nowait=True)
        self.channel_open = True