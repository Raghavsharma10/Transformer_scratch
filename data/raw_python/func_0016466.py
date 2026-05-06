def declare_consumer(self, queue, no_ack, callback, consumer_tag,
            nowait=False):
        """Declare a consumer."""
        return self.channel.basic_consume(queue=queue,
                                          no_ack=no_ack,
                                          callback=callback,
                                          consumer_tag=consumer_tag)