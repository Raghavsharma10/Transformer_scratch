def set_qos(self, prefetch_size=0, prefetch_count=0, apply_globally=False):
        """
        Specify quality of service by requesting that messages be pre-fetched
        from the server. Pre-fetching means that the server will deliver messages
        to the client while the client is still processing unacknowledged messages.

        This method is a :ref:`coroutine <coroutine>`.

        :param int prefetch_size: Specifies a prefetch window in bytes.
            Messages smaller than this will be sent from the server in advance.
            This value may be set to 0, which means "no specific limit".

        :param int prefetch_count: Specifies a prefetch window in terms of whole messages.

        :param bool apply_globally: If true, apply these QoS settings on a global level.
            The meaning of this is implementation-dependent. From the
            `RabbitMQ documentation <https://www.rabbitmq.com/amqp-0-9-1-reference.html#basic.qos.global>`_:

                RabbitMQ has reinterpreted this field. The original specification said:
                "By default the QoS settings apply to the current channel only.
                If this field is set, they are applied to the entire connection."
                Instead, RabbitMQ takes global=false to mean that the QoS settings should apply
                per-consumer (for new consumers on the channel; existing ones being unaffected) and
                global=true to mean that the QoS settings should apply per-channel.
        """
        self.sender.send_BasicQos(prefetch_size, prefetch_count, apply_globally)
        yield from self.synchroniser.wait(spec.BasicQosOK)
        self.reader.ready()