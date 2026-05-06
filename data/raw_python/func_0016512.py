def declare_consumer(self, queue, no_ack, callback, consumer_tag,
            nowait=False):
        """Declare a consumer."""

        @functools.wraps(callback)
        def _callback_decode(channel, method, header, body):
            return callback((channel, method, header, body))

        return self.channel.basic_consume(_callback_decode,
                                          queue=queue,
                                          no_ack=no_ack,
                                          consumer_tag=consumer_tag)