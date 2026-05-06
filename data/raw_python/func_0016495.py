def get(self, *args, **kwargs):
        """Get the next waiting message from the queue.

        :returns: A :class:`Message` instance, or ``None`` if there is
            no messages waiting.

        """
        if not mqueue.qsize():
            return None
        message_data, content_type, content_encoding = mqueue.get()
        return self.Message(backend=self, body=message_data,
                       content_type=content_type,
                       content_encoding=content_encoding)