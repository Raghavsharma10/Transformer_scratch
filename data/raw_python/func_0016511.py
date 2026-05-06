def queue_declare(self, queue, durable, exclusive, auto_delete,
            warn_if_exists=False, arguments=None):
        """Declare a named queue."""

        return self.channel.queue_declare(queue=queue,
                                          durable=durable,
                                          exclusive=exclusive,
                                          auto_delete=auto_delete,
                                          arguments=arguments)