def fetch(self, no_ack=None, auto_ack=None, enable_callbacks=False):
        """Receive the next message waiting on the queue.

        :returns: A :class:`carrot.backends.base.BaseMessage` instance,
            or ``None`` if there's no messages to be received.

        :keyword enable_callbacks: Enable callbacks. The message will be
            processed with all registered callbacks. Default is disabled.
        :keyword auto_ack: Override the default :attr:`auto_ack` setting.
        :keyword no_ack: Override the default :attr:`no_ack` setting.

        """
        no_ack = no_ack or self.no_ack
        auto_ack = auto_ack or self.auto_ack
        message = self.backend.get(self.queue, no_ack=no_ack)
        if message:
            if auto_ack and not message.acknowledged:
                message.ack()
            if enable_callbacks:
                self.receive(message.payload, message)
        return message