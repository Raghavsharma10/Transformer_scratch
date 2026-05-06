def send(self, message_data, routing_key=None, delivery_mode=None,
            mandatory=False, immediate=False, priority=0, content_type=None,
            content_encoding=None, serializer=None, exchange=None):
        """Send a message.

        :param message_data: The message data to send. Can be a list,
            dictionary or a string.

        :keyword routing_key: A custom routing key for the message.
            If not set, the default routing key set in the :attr:`routing_key`
            attribute is used.

        :keyword mandatory: If set, the message has mandatory routing.
            By default the message is silently dropped by the server if it
            can't be routed to a queue. However - If the message is mandatory,
            an exception will be raised instead.

        :keyword immediate: Request immediate delivery.
            If the message cannot be routed to a queue consumer immediately,
            an exception will be raised. This is instead of the default
            behaviour, where the server will accept and queue the message,
            but with no guarantee that the message will ever be consumed.

        :keyword delivery_mode: Override the default :attr:`delivery_mode`.

        :keyword priority: The message priority, ``0`` to ``9``.

        :keyword content_type: The messages content_type. If content_type
            is set, no serialization occurs as it is assumed this is either
            a binary object, or you've done your own serialization.
            Leave blank if using built-in serialization as our library
            properly sets content_type.

        :keyword content_encoding: The character set in which this object
            is encoded. Use "binary" if sending in raw binary objects.
            Leave blank if using built-in serialization as our library
            properly sets content_encoding.

        :keyword serializer: Override the default :attr:`serializer`.

        :keyword exchange: Override the exchange to publish to.
            Note that this exchange must have been declared.

        """
        headers = None
        routing_key = routing_key or self.routing_key

        if self.exchange_type == "headers":
            headers, routing_key = routing_key, ""

        exchange = exchange or self.exchange

        message = self.create_message(message_data, priority=priority,
                                      delivery_mode=delivery_mode,
                                      content_type=content_type,
                                      content_encoding=content_encoding,
                                      serializer=serializer)
        self.backend.publish(message,
                             exchange=exchange, routing_key=routing_key,
                             mandatory=mandatory, immediate=immediate,
                             headers=headers)