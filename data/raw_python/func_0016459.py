def get(self, queue, no_ack=False):
        """Receive a message from a declared queue by name.

        :returns: A :class:`Message` object if a message was received,
            ``None`` otherwise. If ``None`` was returned, it probably means
            there was no messages waiting on the queue.

        """
        raw_message = self.channel.basic_get(queue, no_ack=no_ack)
        if not raw_message:
            return None
        return self.message_to_python(raw_message)