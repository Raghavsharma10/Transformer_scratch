def publish(self, message, routing_key, *, mandatory=True):
        """
        Publish a message on the exchange, to be asynchronously delivered to queues.

        :param asynqp.Message message: the message to send
        :param str routing_key: the routing key with which to publish the message
        :param bool mandatory: if True (the default) undeliverable messages result in an error (see also :meth:`Channel.set_return_handler`)
        """
        self.sender.send_BasicPublish(self.name, routing_key, mandatory, message)