def send(self, message_data, delivery_mode=None):
        """See :meth:`Publisher.send`"""
        self.publisher.send(message_data, delivery_mode=delivery_mode)