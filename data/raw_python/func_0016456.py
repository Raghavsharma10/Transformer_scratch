def queue_declare(self, queue, durable, exclusive, auto_delete,
            warn_if_exists=False, arguments=None):
        """Declare a named queue."""
        if warn_if_exists and self.queue_exists(queue):
            warnings.warn(QueueAlreadyExistsWarning(
                QueueAlreadyExistsWarning.__doc__))

        return self.channel.queue_declare(queue=queue,
                                          durable=durable,
                                          exclusive=exclusive,
                                          auto_delete=auto_delete,
                                          arguments=arguments)