def queue_bind(self, queue, exchange, routing_key, arguments=None):
        """Bind queue to an exchange using a routing key."""
        return self.channel.queue_bind(queue=queue,
                                       exchange=exchange,
                                       routing_key=routing_key,
                                       arguments=arguments)