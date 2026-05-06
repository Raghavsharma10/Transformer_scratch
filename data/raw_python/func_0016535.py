def _declare_consumer(self, consumer, nowait=False):
        """Declare consumer so messages can be received from it using
        :meth:`iterconsume`."""
        if consumer.queue not in self._open_consumers:
            # Use the ConsumerSet's consumer by default, but if the
            # child consumer has a callback, honor it.
            callback = consumer.callbacks and \
                consumer._receive_callback or self._receive_callback
            self.backend.declare_consumer(queue=consumer.queue,
                                          no_ack=consumer.no_ack,
                                          nowait=nowait,
                                          callback=callback,
                                          consumer_tag=consumer.consumer_tag)
            self._open_consumers[consumer.queue] = consumer.consumer_tag