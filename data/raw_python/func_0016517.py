def declare(self):
        """Declares the queue, the exchange and binds the queue to
        the exchange."""
        arguments = None
        routing_key = self.routing_key
        if self.exchange_type == "headers":
            arguments, routing_key = routing_key, ""

        if self.queue:
            self.backend.queue_declare(queue=self.queue, durable=self.durable,
                                       exclusive=self.exclusive,
                                       auto_delete=self.auto_delete,
                                       arguments=self.queue_arguments,
                                       warn_if_exists=self.warn_if_exists)
        if self.exchange:
            self.backend.exchange_declare(exchange=self.exchange,
                                          type=self.exchange_type,
                                          durable=self.durable,
                                          auto_delete=self.auto_delete)
        if self.queue:
            self.backend.queue_bind(queue=self.queue,
                                    exchange=self.exchange,
                                    routing_key=routing_key,
                                    arguments=arguments)
        self._closed = False
        return self